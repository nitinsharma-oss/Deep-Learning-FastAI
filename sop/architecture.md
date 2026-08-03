# NFC Digital Identity Platform
## Step 1 — System Architecture

Version 0.1 · architecture draft · no code

This document covers service topology, trust boundaries, the resolver hot path, the ownership
model, the chip abstraction, and the event pipeline. It stops short of column-level schema —
that is Step 2 (Database Design). Where a decision constrains the schema, it is stated here and
marked so Step 2 can implement it directly.

---

## 0. Contradictions resolved

Three requirements in the brief conflict with each other. Each is resolved below rather than
silently reinterpreted later.

### 0.1 "Cards belong to Profiles" vs enterprise ownership

The brief states both `Account → Profiles → Cards` and `Organization → Accounts → Profiles →
Cards`, and separately requires that organisations can assign, revoke, and transfer cards. These
cannot all hold. If a card belongs to a profile, and the profile belongs to an employee's
account, then a company cannot revoke a card it paid for without reaching through an account it
does not control.

**Resolution — a card carries three independent relationships:**

| Edge | Meaning | Points at |
| --- | --- | --- |
| `owner` | Legal title. Who bought it, who can revoke or reassign it. | Organization **or** account, never both |
| `holder` | Custody. Whose card this currently is. | Account |
| `profile` | Resolution target. What a tap serves. | Profile |

For a consumer, owner and holder are the same account and the distinction is invisible. For an
enterprise, the organisation owns and the employee holds. When an employee leaves, the
organisation revokes the card and holder is cleared — the employee keeps their profile and their
personal URL, and the organisation keeps the asset. That outcome is only expressible with the
edges separated.

The integrity invariant survives in a stronger form: **the profile a card resolves to must belong
to the account holding it.** That is a composite foreign key, enforced in Postgres, not
application code:

```
profiles  UNIQUE (id, account_id)
cards     FOREIGN KEY (profile_id, holder_account_id)
              REFERENCES profiles (id, account_id)
```

Owner is then orthogonal and enforced separately:

```
cards     CHECK (num_nonnulls(owner_org_id, owner_account_id) = 1)
```

A card can never point at a stranger's profile, and can never be simultaneously owned by a
company and a person.

### 0.2 "The resolver must never verify subscriptions" vs lapsed accounts

The brief forbids the resolver from checking subscriptions or touching Stripe. It also implies
that a lapsed subscription should stop serving. Taken literally together, a customer who cancels
keeps a working card forever.

**Resolution — precompute, don't check.** The card row carries a single denormalised
`serving_state` enum, and the resolver reads it from the row it has already loaded. No join, no
Stripe call, no business logic on the hot path. A background worker recomputes `serving_state`
whenever a subscription, card status, profile status, or organisation status changes, and
invalidates the cache key for that token.

The resolver's entire decision becomes one comparison against a column it already has. Both
requirements hold simultaneously.

### 0.3 Four analytics tables vs one event per tap

The brief specifies `tap_events`, `qr_events`, `device_events`, `geo_events`. Splitting by
**source** is sound — a QR scan and an NFC tap are genuinely different acquisition channels.
Splitting by **attribute** is not: device and geography are properties *of* a tap, not separate
events. Four tables means four writes and a four-way join to answer "taps from Android in India
last week."

**Resolution — split by source, keep attributes as columns.** One partitioned `tap_events` table
with `channel` (nfc / qr / direct), plus device and geo columns resolved at ingest. If a genuine
second event stream appears later (profile shares, link clicks), it gets its own table then.

This is a recommendation, not a resolution — flag it if you want the four-table split retained
for a reason not in the brief.

---

## 1. Architectural principles

These are the load-bearing ones. Everything below follows from them.

**The card is a pointer, not a credential.** A 128-bit random token, base62-encoded, unique,
indexed, non-enumerable. Nothing else is ever written to a chip. Compromise of a token yields at
most a redirect to a public page that was already public.

**Read path and write path are different systems.** The resolver is stateless, cacheable, and
sized for millions of requests. Everything else is a conventional authenticated application
sized for thousands. They share a database and nothing else. A card going viral must not slow
down profile editing, and profile editing must never be able to take down resolution.

**Identity is permanent; tokens are disposable.** The 302 indirection means a card can be
replaced without the profile URL ever changing. This is also why permanent chip locking costs
nothing (§5).

**Integrity lives in Postgres.** Constraints, composite foreign keys, check constraints, and
append-only enforcement. Application validation is a usability layer that produces good error
messages; it is not the guarantee.

**Nothing important is mutable.** Ownership transfers, activations, encodings, and admin actions
are append-only. Current state is derived or denormalised from the log, never the only record.

---

## 2. Service topology

Five deployable units. The split is by traffic profile and blast radius, not by domain
fashion — a resolver outage takes down every card in the field, so it shares as little as
possible with anything else.

| Service | Traffic | Scales on | Failure impact |
| --- | --- | --- | --- |
| **Resolver** | Very high, read-only | Taps | Every card stops working |
| **Profile renderer** | High, read-mostly | Profile views | Profiles stop rendering; cards still redirect |
| **Application API** | Low, read/write | Logged-in users | Dashboard down; taps unaffected |
| **Admin API** | Very low, high privilege | Staff | Approvals paused |
| **Workers** | Async | Event volume | Analytics lag; nothing user-facing breaks |

The Admin API is a separate deployable specifically so it can sit behind different network
controls, a different auth policy, and mandatory step-up authentication. It is the only surface
that can approve an ownership transfer, which makes it the highest-value target in the system.

### Trust zones

- **Untrusted** — the chip, the tap URL, the visitor's browser. Assume attacker-controlled.
- **Public** — resolver and profile renderer. No authentication, no PII writes, read-only
  against a replica where possible.
- **Authenticated** — application API. Session-bound, rate-limited per account.
- **Privileged** — admin API. Staff identity, step-up auth, every action audited.
- **Internal** — workers, database, object storage. No public ingress.

---

## 3. The resolver

The single most important component. Everything about it is a consequence of "must stay fast
under millions of requests while remaining correct about revocation."

### Request path

```
GET /t/{token}
  → validate token shape (length, charset)         reject malformed before any I/O
  → cache lookup by token                          Redis, hot path
      hit  → serving_state + slug
      miss → single indexed row read from Postgres, populate cache
  → evaluate serving_state
      active   → 302 to /u/{slug}
      revoked  → 410 Gone, static explanation page
      inactive → 404, or claim page if provisioned
      lapsed   → 302 to a "profile paused" page
  → emit tap event to the queue, fire-and-forget
  → respond
```

Everything the resolver needs is on one row: `token`, `serving_state`, `slug`. Denormalising
`slug` onto the card row removes the last join. The write path keeps it consistent; the read path
never derives it.

### Cache policy

Cache the resolution, not the card. Key on token, value is `{serving_state, slug}`, TTL in
minutes with explicit invalidation on any state change. A short TTL bounds the damage if an
invalidation is ever missed — a revoked card that keeps working for ten minutes is a bug; one
that keeps working forever is a breach.

**Revocation must be immediate**, so the resolver response carries `Cache-Control: no-store` and
returns **302, never 301**. A 301 is cached by browsers indefinitely and cannot be recalled —
adopting one would make revocation permanently impossible for every device that ever saw it.
This is worth a comment in the code and a test that asserts the status code.

### What the resolver must never do

No subscription checks, no Stripe, no page generation, no joins, no synchronous analytics
writes, no business logic. If a feature request would add any of these, it belongs in the worker
or the renderer.

---

## 4. Ownership and lifecycle

### Entities (conceptual — schema in Step 2)

```
organizations ──< org_memberships >── accounts ──< profiles ──< (resolution target)
                                          │            │
                                          └── holds ───┼──< cards >── owned by org or account
                                                       │
                                          slug_history ┘
```

- **organizations** — enterprise customers. Optional; consumer accounts have none.
- **accounts** — a person. The unit of authentication and of consumer billing.
- **org_memberships** — account ↔ organisation with an org-scoped role.
- **profiles** — an identity. An account may hold several. Owns the permanent slug.
- **slug_history** — every slug a profile has ever had, so old links redirect forever.
- **cards** — the physical object. Carries the three edges from §0.1.
- **ownership_transfers** — append-only title record.

### Card states

`manufactured → provisioned → claimed → active → (suspended) → revoked`

Only `active` resolves. `revoked` is terminal — a revoked token is never reused, never
reactivated, and returns 410 forever.

### Activation

A manufactured card must not become active by being tapped. The claim secret is a scratch code
printed on the packaging, stored only as a hash on the card row:

```
POST /cards/claim      { token, claim_code }   → binds holder, requires authenticated account
POST /cards/activate   { card_id }             → transitions to active, writes audit + transfer
POST /cards/revoke     { card_id, reason }     → terminal, owner or admin only
```

Claim is rate-limited per token *and* per account, with lockout after a small number of failures —
a scratch code is short enough to brute-force otherwise. First activation writes the first
`ownership_transfers` row with a null `from`.

For enterprise, claim is inverted: the organisation bulk-claims at purchase, then assigns to
employees. Assignment changes holder, not owner, and does not write a transfer row — it writes
an assignment audit event. Only title changes are transfers.

### Transfers

The only path to a new owner runs through admin approval. A transfer request is a row in a
pending state; approval writes the append-only record, updates the denormalised
`cards.owner_*`, invalidates cache, and emits an audit event — in one transaction. Nothing about
the previous ownership is overwritten.

---

## 5. Chip abstraction

Two chip profiles must coexist without a resolver rewrite, so token verification is a strategy
selected by a `chip_profile` column, not an `if` statement.

| Profile | Chip | Verification | Anti-clone |
| --- | --- | --- | --- |
| **Basic** | NTAG213/215/216 | Token exists, is active | No — a copied token works |
| **Enterprise** | NTAG424 DNA | Token plus AES-CMAC over rotating PICC data | Yes — a replayed tap is detectable |

Both arrive at the same endpoint. Basic taps carry `/t/{token}`; DNA taps carry
`/t/{token}?picc_data=…&cmac=…`. The resolver dispatches on `chip_profile`, and the DNA path adds
one AES verification and a monotonic counter check — both cheap, neither requiring a second round
trip. Designing this as a strategy now is what makes "adopt stronger chips later" true rather
than aspirational.

DNA key material lives in a KMS or HSM, never in the application database, and never in the same
blast radius as the token table.

### Locking is mandatory, and it is free

Because the token never changes over a card's life — reissue mints a *new* token on a *new* card
and repoints it at the same permanent slug — there is no operational reason to ever rewrite a
chip in the field. Permanent locking therefore forfeits nothing.

Locking happens at the encoding station before shipping, and is recorded as an immutable audit
event carrying encoder identity, station, timestamp, chip profile, and a post-write verification
read. `chip_locked` as a mutable boolean is explicitly rejected: a flag can be flipped by a bug,
an audit event cannot be un-written.

The threat this closes is not account takeover — the registry already prevents that. It is
**phishing by rewrite**: a card in a victim's wallet re-pointed at an attacker's page, handed
out by the victim to people who trust them. That attack survives every server-side control in
this document, and locking is the only thing that stops it.

---

## 6. Event and analytics pipeline

The resolver emits and forgets. Nothing downstream can slow a redirect or fail one.

```
resolver → queue → ingest worker → tap_events (partitioned monthly)
                                 → rollup worker → daily aggregates → dashboard
```

- **Emit** is fire-and-forget to a queue. If the queue is unavailable, the tap still redirects
  and the event is dropped. Analytics are best-effort; resolution is not.
- **Ingest** resolves geo and device from the request context, and writes one row.
- **Rollups** precompute the dashboard's daily and weekly aggregates. The dashboard never queries
  the raw event table.
- **Partitioning** is monthly on `occurred_at`, so retention is a partition drop rather than a
  mass delete.

### Privacy

`tap_events` is visitor data — people who never agreed to anything. Under GDPR the platform is
controller for it. Therefore: no raw IP storage (derive country at ingest, discard the address),
truncated user-agent to a device class rather than a full fingerprint, a hard retention window
after which partitions are dropped, and aggregates that survive the raw rows.

This table will exceed every other table combined within months. Treat its growth as a design
input, not an operational surprise.

---

## 7. Identity, sessions, and RBAC

### Authentication

Email plus password, then OTP. Sessions are opaque server-side records, not self-contained JWTs,
so revocation is immediate — the Security page promises "connected devices" and "sign out
everywhere," and only server-side sessions can honour that. Refresh token rotation with reuse
detection: a replayed refresh token invalidates the whole family and raises a security event.

Cookies are `HttpOnly`, `Secure`, `SameSite=Lax`, with CSRF tokens on state-changing requests.

### Two role scopes

The brief lists Employee, Manager, HR, Organization Admin, Super Admin as one list. They are not
one list — four are scoped to an organisation, one is scoped to the platform. Conflating them is
how privilege escalation bugs happen.

| Scope | Roles | Stored on |
| --- | --- | --- |
| Platform | `support`, `admin`, `super_admin` | `accounts.platform_role` |
| Organisation | `employee`, `manager`, `hr`, `org_admin` | `org_memberships.role` |

An org admin is powerful inside their organisation and has no platform privileges whatsoever.
Every authorisation check names its scope explicitly; there is no ambient "is admin" boolean.

### Step-up

Approving a transfer, revoking a card, and changing organisation ownership require re-
authentication within a short window regardless of session age.

---

## 8. Audit architecture

Two append-only records with a clear boundary, because they answer different questions.

- **`ownership_transfers`** — the record of *title*. Who owned this card, when, approved by whom,
  why. This is the legal artifact.
- **`audit_log`** — the record of *actions*. Issuance, encoding, locking, activation, assignment,
  profile edits, admin actions, subscription changes, security events.

A transfer writes to both, in one transaction. Enforcement is structural: `REVOKE UPDATE, DELETE`
on both tables from the application role, so append-only is a database grant rather than a
convention someone can forget.

Audit rows record actor, actor scope, subject, action, before/after where meaningful, request
correlation ID, and timestamp. They are written in the same transaction as the change they
describe — an audit log that can diverge from reality is worse than none.

---

## 9. Deployment topology

```
CDN / edge
  └── Resolver            stateless, horizontally scaled, no public DB access
        ├── Redis         resolution cache
        └── Postgres      read replica for cache misses

Application API           authenticated, rate-limited
Admin API                 privileged, network-restricted, step-up auth
Workers                   queue consumers, no ingress
Postgres primary          writes; replicas for read paths
Object storage            profile media, signed URLs only
Stripe                    webhooks in; never called on a read path
KMS / HSM                 NTAG424 DNA keys
```

Resolver cache misses read a replica. Replica lag is acceptable for resolution because state
changes invalidate the cache explicitly — the miss path is a fallback, not the source of truth
for freshness.

---

## 10. Decisions needing sign-off before Step 2

1. **Three-edge ownership (§0.1)** — this shapes every table. Confirm before schema.
2. **`serving_state` denormalisation (§0.2)** — the alternative is the resolver joining
   subscriptions, which the brief forbids.
3. **One `tap_events` table rather than four (§0.3)** — the only place I'm recommending against
   the brief on a non-security matter.
4. **Multiple profiles per account** — the brief says the account layer enables it. Confirm it is
   in scope now; it affects the composite key and the dashboard's information architecture.
5. **Consumer billing per account, enterprise billing per organisation** — two subscription
   owners, one table with a check constraint, or two tables?
6. **Can an organisation-owned card point at a personal profile?** The composite FK says the
   profile must belong to the holder. If an employer wants control over the *content* an employee's
   card serves, that needs org-owned profiles and it needs deciding now.

---

## Step 2 will contain

Full Prisma-targeted schema: every table, column, type, index, constraint, and cascade rule; the
composite and check constraints from §0.1 written out; partitioning DDL for `tap_events`; the
append-only grants from §8; and the migration order.
