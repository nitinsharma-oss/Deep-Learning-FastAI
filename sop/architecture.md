# NFC Digital Identity Platform
## Step 1 — System Architecture

**Version 0.2** · architecture draft · no code
Supersedes v0.1. Target: 10M users, external security audit before launch.

Optimised for correctness, security, scalability, maintainability and extensibility — in that
order. Where a safer architecture costs implementation time, this document takes the safer one
and states what it costs.

---

## 1. Scale envelope

Architecture without arithmetic is decoration. These are the numbers the design is sized against;
correct them if your assumptions differ, because several decisions below change if they do.

| Quantity | Assumption | Derived |
| --- | --- | --- |
| Accounts | 10,000,000 | — |
| Cards | ~1.2 per account | 12M rows — trivial for Postgres |
| Profiles | ~1.3 per account | 13M rows — trivial |
| Taps per card per month | ~10 | 100M events/month |
| Average resolution rate | — | **~38 req/s** |
| Peak resolution rate | 20× average | **~800 req/s** |
| `tap_events` growth | 100M rows/month | **~1.2B rows/year** |
| Authenticated sessions | 2% of users daily | ~200k sessions/day |

**The conclusion that matters:** resolution is a small problem and analytics is a large one. 800
req/s of single-key cache reads is served by a modest cluster. 1.2 billion rows a year, carrying
visitor personal data subject to erasure requests, is where the genuine engineering is.

Sizing the resolver for imagined millions while treating `tap_events` as an afterthought is the
default failure mode for this shape of product. This document inverts that emphasis.

**The load that actually threatens the resolver is adversarial, not organic** — see §5.3.

---

## 2. Decisions resolved

v0.1 left six decisions open. All are resolved here, choosing the safer option, with the cost of
that choice stated.

### 2.1 Three-edge card ownership — **confirmed**

A card carries three independent relationships: `owner` (title — organisation or account),
`holder` (custody — account), `profile` (resolution target).

*Trade-off:* every ownership query names which edge it means, and the dashboard has to explain
the difference to enterprise admins. Accepted, because collapsing them makes employer revocation
inexpressible and there is no later migration that adds the distinction cheaply.

### 2.2 `serving_state` denormalised onto the card — **confirmed, with reconciliation**

The resolver reads one precomputed column. But a denormalised value is a cache, and caches drift:
a worker crash between a subscription webhook and the state recompute leaves a card serving when
it should not, silently and indefinitely.

**Added:** a reconciliation job recomputes `serving_state` from source truth for every card on a
rolling window, and emits a metric for every mismatch found. A non-zero drift count is an alarm,
not a statistic.

*Trade-off:* a background job that reads every card row periodically. At 12M rows that is minutes
of replica load per cycle. Cheap relative to serving a cancelled customer's card for a year.

### 2.3 One partitioned `tap_events` table, not four — **confirmed**

Split by *source* (`channel`: nfc / qr / direct), not by *attribute*. Device and geography are
columns on the event.

*Trade-off:* diverges from the brief. If the four-table split exists to route different event
types to different retention policies, say so and I will reinstate it — that would be a real
reason. Volume alone is not, because partitioning solves volume and joins do not.

### 2.4 Multiple profiles per account — **in scope now**

*Trade-off:* the dashboard gains a profile-switcher and every profile-scoped query gains a
selector. Real but contained cost. The alternative — retrofitting later — is a breaking change to
the composite keys in §6, the URL structure, and every authorisation check. Doing it now is
roughly a tenth of the cost of doing it later.

### 2.5 Billing: one `subscriptions` table with a dual-owner constraint

Mirrors the card ownership pattern exactly: nullable `owner_account_id` and `owner_org_id`, with
`CHECK (num_nonnulls(...) = 1)`.

*Trade-off:* one table serving two billing models means every query filters on owner type. The
gain is pattern consistency — an auditor learning the ownership idiom once can verify it
everywhere, and one set of constraints is easier to prove correct than two divergent tables.

### 2.6 Can an org-owned card point at a personal profile? — **No**

This was the decision I most wanted your input on. Resolved as: **organisation-owned cards must
point at organisation-owned profiles.** Which means organisations own profiles too, not just
cards.

The reasoning is that the alternative is unsafe in both directions. If a company's card resolves
to an employee's personal profile, the employee can change what the company's card says at any
moment, and the company's brand is on the physical object. Run it the other way and on
termination there is a live dispute over a URL that is simultaneously the employee's identity and
the employer's asset — with no principled answer, because both claims are real.

Separating them makes both claims true at once: the employee has a personal profile at a
permanent personal URL that is theirs forever, and the organisation has a work profile on a work
card that stays with the organisation.

*Trade-off, stated plainly:* an employee has two profiles and two URLs, which is more to explain
and more to build. And the "permanent identity" promise becomes conditional — permanent for
personal profiles, tied to employment for work profiles. That nuance must appear in the product
copy, not just the schema, or it becomes a support burden and a trust problem the first time
someone loses a work URL they had been handing out for three years.

---

## 3. Principles

**The card is a pointer, not a credential.** 128 bits of CSPRNG entropy, base62, unique, indexed,
non-enumerable. Nothing else is ever written to a chip. A leaked token yields a redirect to a page
that was already public.

**Read path and write path are separate systems.** The resolver shares a database with the
application and nothing else. A viral card must not slow profile editing; profile editing must
never be able to break resolution.

**Identity is permanent; tokens are disposable.** The 302 indirection means a card is replaceable
without the profile URL changing. This is also what makes permanent chip locking free (§7.2).

**Integrity is enforced in Postgres.** Composite foreign keys, check constraints, and privilege
grants. Application validation exists to produce good error messages; it is not the guarantee. An
auditor can read a constraint. An auditor cannot read every code path that might write to a table.

**Nothing important is mutable.** Transfers, activations, encodings and admin actions are
append-only, enforced by `REVOKE UPDATE, DELETE` rather than by convention.

**Every control must be demonstrable.** If a security property cannot be shown to an auditor as a
constraint, a grant, a test, or a metric, it is an intention rather than a control.

---

## 4. Service topology

Five deployable units, split by traffic profile and blast radius.

| Service | Traffic | Failure impact |
| --- | --- | --- |
| **Resolver** | ~800 req/s peak, read-only | Every card in the field stops working |
| **Profile renderer** | High, read-mostly | Profiles blank; cards still redirect |
| **Application API** | ~200k sessions/day | Dashboard down; taps unaffected |
| **Admin API** | Very low, highest privilege | Approvals paused |
| **Workers** | 100M events/month | Analytics lag; nothing user-facing breaks |

The Admin API is separately deployable specifically so it can sit behind different network
controls, mandatory step-up authentication, and its own audit pipeline. It is the only surface
that can approve an ownership transfer, which makes it the highest-value target in the system.

### Trust zones

| Zone | Contains | Assumption |
| --- | --- | --- |
| Untrusted | Chip, tap URL, visitor browser | Attacker-controlled |
| Public | Resolver, profile renderer | No auth, no PII writes, replica reads |
| Authenticated | Application API | Session-bound, per-account rate limits |
| Privileged | Admin API | Staff identity, step-up, fully audited |
| Internal | Workers, database, object storage, KMS | No public ingress |

---

## 5. The resolver

### 5.1 Request path

```
GET /t/{token}
  → shape validation                  reject malformed before any I/O
  → negative cache check              known-bad tokens die here (§5.3)
  → resolution cache (Redis)
      hit  → {serving_state, slug}
      miss → single-flight fetch from Postgres replica, populate
  → evaluate serving_state
      active    → 302 → /u/{slug}
      lapsed    → 302 → paused page
      revoked   → 410 Gone
      unclaimed → 404, or claim page
  → emit tap event to local buffer, non-blocking
  → respond
```

Everything needed sits on one row: `token`, `serving_state`, `slug`. `slug` is denormalised onto
the card row to remove the last join; the write path keeps it consistent.

### 5.2 302, never 301

A 301 is cached by browsers indefinitely and cannot be recalled. Adopting one would make
revocation permanently impossible on every device that had ever seen the card — a defect with no
remedy short of physically retrieving the card.

This gets an explicit test asserting the status code, and a comment explaining why, because it is
exactly the kind of thing a well-meaning performance optimisation changes later.

### 5.3 Negative caching and the amplification DoS

**This is the resolver's real threat.** Organic load is ~800 req/s at peak. An attacker
generating random tokens produces a 100% miss rate: every request bypasses Redis and reaches
Postgres. Modest attacker bandwidth converts into database saturation on the one service whose
failure takes down every card in the field.

Three controls, layered:

1. **Shape validation before I/O.** Wrong length or charset is rejected without touching Redis or
   Postgres. Eliminates unsophisticated floods for free.
2. **Negative caching.** Tokens confirmed absent are cached as absent, with a shorter TTL than
   positive entries. Repeat attempts terminate in Redis.
3. **Edge rate limiting on cache-miss rate per source**, not on request rate. Legitimate traffic
   is nearly all hits; a source with a high miss ratio is enumerating.

*Trade-off:* negative caching means a newly provisioned token can 404 for up to its negative TTL
after creation. Mitigated by explicitly purging the negative entry at provisioning time, and by
keeping negative TTL short.

### 5.4 Cache stampede

A card featured somewhere popular produces thousands of concurrent requests for one uncached
token. Without coordination, all of them miss and all of them query. Resolution is
**single-flight**: one fetch per key, concurrent callers wait on it.

### 5.5 Event emission must not be a dependency

"Fire and forget" is a lie if it is a synchronous network call — a slow queue becomes resolver
latency, and an unavailable queue becomes resolver failure. Events are written to a bounded
in-process buffer and flushed in batches by a background task. **A full buffer drops events.**

*Trade-off:* analytics lose events during incidents and deploys — exactly when volume is most
interesting. Accepted deliberately: resolution is a correctness guarantee, analytics are
best-effort, and no analytics feature is worth a redirect that fails.

### 5.6 Forbidden on the hot path

No subscription checks. No Stripe. No page rendering. No joins. No synchronous writes. No
business logic. If a feature needs any of these, it belongs in a worker or the renderer.

---

## 6. Ownership model

### Entities

```
organizations ──< org_memberships >── accounts
       │                                  │
       └──────────< profiles >────────────┘         profile owned by org OR account
                       │
                  slug_history                       every slug ever, forever
                       │
                    cards                            owner · holder · profile
                       │
              ownership_transfers                    append-only title record
```

### The invariants, enforced as constraints

The integrity rules are three composite foreign keys plus one check. All use Postgres's default
`MATCH SIMPLE`, under which a constraint containing a NULL column is not enforced — so each key
activates only for the ownership shape it governs, with no triggers and no application logic.

```
-- exactly one owner
CHECK (num_nonnulls(owner_account_id, owner_org_id) = 1)

-- personal card: profile must belong to the holder
FOREIGN KEY (profile_id, holder_account_id)
    REFERENCES profiles (id, owner_account_id)

-- org card: profile must belong to the owning organisation
FOREIGN KEY (profile_id, owner_org_id)
    REFERENCES profiles (id, owner_org_id)

-- org card: holder must be a member of the owning organisation
FOREIGN KEY (holder_account_id, owner_org_id)
    REFERENCES org_memberships (account_id, org_id)
```

A personal card leaves `owner_org_id` NULL, so the two org constraints stand down. An org card
leaves `owner_account_id` NULL, so the personal constraint stands down. Every ownership shape is
covered, and an invalid one cannot be inserted by any code path — including a migration script,
a console session, or a bug.

This is the single most audit-legible thing in the design. It is worth the modest schema
awkwardness of nullable owner columns.

### Card lifecycle

`manufactured → provisioned → claimed → active → (suspended) → revoked`

Only `active` resolves. `revoked` is terminal: the token is never reused, never reactivated, and
returns 410 permanently.

### Activation

A manufactured card must not activate by being tapped. A scratch code on the packaging is stored
only as a hash on the card row.

```
POST /cards/claim      { token, claim_code }   authenticated; binds holder
POST /cards/activate   { card_id }             writes audit + first transfer row
POST /cards/revoke     { card_id, reason }     terminal; owner or admin
```

Security requirements on claim, all of which an auditor will test:

- **Uniform responses.** "Unknown token" and "wrong code" return identical bodies, status codes
  and timing. Otherwise the endpoint is a card-existence oracle.
- **Constant-time comparison** of the code hash.
- **Rate limited per token and per account**, with lockout. A scratch code humans can type is
  short; only rate limiting closes the gap.
- **Slow KDF** (Argon2id) on the claim hash, so a database disclosure does not yield a
  bulk-crackable set of activation codes.
- **Single use**, invalidated inside the activation transaction.

*Trade-off:* Argon2id makes claim measurably slower. Claim happens once per card in its lifetime.
Irrelevant cost.

For enterprise the flow inverts: the organisation bulk-claims at purchase, then assigns to
employees. **Assignment changes holder, not owner**, and writes an assignment audit event rather
than a transfer row. Only changes of title are transfers.

### Transfers

The only edge into new ownership runs through admin approval. Approval writes the append-only
transfer row, updates the denormalised owner columns, invalidates cache, and emits an audit event
— in one transaction. Nothing about prior ownership is overwritten.

---

## 7. Chip abstraction

### 7.1 Two profiles, one endpoint

Token verification is a strategy selected by a `chip_profile` column, not a branch bolted on
later.

| Profile | Chip | Verification | Clone-resistant |
| --- | --- | --- | --- |
| Basic | NTAG213/215/216 | Token exists and is active | No |
| Enterprise | NTAG424 DNA | Token + AES-CMAC over rotating PICC data | Yes |

Basic taps arrive as `/t/{token}`; DNA taps as `/t/{token}?picc_data=…&cmac=…`. The DNA path adds
one AES verification and a monotonic counter check — both in-process, neither adding a round
trip. Designing the strategy now is what makes "adopt stronger chips later" true rather than
aspirational.

### 7.2 Key diversification is mandatory

**Never one shared AES key across the fleet.** A single extracted key from one chip in an
attacker's hands would compromise all 10M cards, and chip key extraction is a funded, documented
discipline.

Each card gets a key diversified from a master key held in an HSM or KMS, derived per card. The
master key never leaves the HSM and never exists in application memory or the application
database. Compromise of one chip yields one chip.

*Trade-off:* the encoding station needs authenticated HSM access, which makes manufacturing more
operationally complex and harder to outsource. This is not negotiable — a shared fleet key is the
kind of finding that stops a launch.

### 7.3 Locking is mandatory and it is free

The token never changes over a card's life: reissue mints a new token on a new card and repoints
it at the same permanent slug. There is no operational reason to rewrite a chip in the field, so
permanent locking forfeits nothing.

Locking happens at the encoding station before shipping and is recorded as an **immutable audit
event** carrying encoder identity, station, timestamp, chip profile, and a post-write verification
read. A mutable `chip_locked` boolean is explicitly rejected — a flag can be flipped by a bug, an
audit event cannot be un-written.

The threat this closes is not account takeover; the registry already prevents that. It is
**phishing by rewrite** — a card in a victim's wallet re-pointed at an attacker's page, then
handed out by the victim to people who trust them. That attack defeats every server-side control
in this document, and locking is the only thing that stops it.

---

## 8. Event pipeline and privacy

```
resolver → in-process buffer → queue → ingest worker → tap_events (monthly partitions)
                                                     → rollup worker → aggregates → dashboard
```

- **Ingest** derives country and device class from request context, then discards the raw inputs.
- **Rollups** precompute dashboard aggregates. The dashboard never queries raw events.
- **Partitioning** is monthly on `occurred_at`, so retention is a partition drop rather than a
  1.2-billion-row delete.

### Privacy posture

`tap_events` records people who never agreed to anything — visitors, not customers. The platform
is data controller for it.

- **No raw IP storage.** Country derived at ingest, address discarded. Never written to disk.
- **Device class, not fingerprint.** "Android phone", not a full user-agent string.
- **Hard retention window**, enforced by dropping partitions on a schedule, not by a policy
  document.
- **Aggregates outlive raw rows**, so the dashboard keeps working after raw events are dropped.

*Trade-off:* per-visitor journey analysis becomes impossible. That is the correct outcome. The
alternative is building a tracking database on non-consenting third parties and hoping the audit
does not notice.

---

## 9. Identity, sessions, RBAC

### Sessions are server-side records, not JWTs

The Security page promises connected devices and sign-out-everywhere. Only server-side sessions
can honour that: a self-contained JWT remains valid until expiry no matter what the server thinks.

*Trade-off:* a datastore read per authenticated request. At 200k sessions/day this is
inconsequential, and it buys immediate revocation — which is the difference between a
compromised-session incident lasting seconds and lasting an hour.

Refresh token rotation with **reuse detection**: a replayed refresh token invalidates the entire
token family and raises a security event. Cookies are `HttpOnly`, `Secure`, `SameSite=Lax`, with
CSRF tokens on state-changing requests.

### Two role scopes, never one list

The brief lists Employee, Manager, HR, Org Admin, Super Admin together. Four are organisation-
scoped, one is platform-scoped. Conflating them is the classic route to privilege escalation.

| Scope | Roles | Stored on |
| --- | --- | --- |
| Platform | `support`, `admin`, `super_admin` | `accounts.platform_role` |
| Organisation | `employee`, `manager`, `hr`, `org_admin` | `org_memberships.role` |

An org admin is powerful inside one organisation and has no platform privileges of any kind.
Every authorisation check names its scope explicitly. There is no ambient "is admin" boolean
anywhere in the system.

### Step-up authentication

Approving a transfer, revoking a card, changing organisation ownership, and any bulk operation
require re-authentication within a short window regardless of session age.

### Idempotency

All state-changing endpoints accept an idempotency key. At this scale retries are routine, and
a duplicated transfer approval or double-charged subscription is both a correctness bug and an
audit finding.

---

## 10. Audit architecture

Two append-only records with a clear boundary:

- **`ownership_transfers`** — the record of *title*. The legal artifact.
- **`audit_log`** — the record of *actions*: issuance, encoding, locking, activation, assignment,
  profile edits, admin actions, subscription changes, security events.

A transfer writes both, in one transaction. Append-only is enforced by `REVOKE UPDATE, DELETE`
from the application role — a database grant, not a convention someone can forget under deadline.

Audit rows carry actor, actor scope, subject, action, before/after where meaningful, request
correlation ID, and timestamp. They are written in the same transaction as the change they
describe; an audit log that can diverge from reality is worse than no audit log, because it is
trusted.

### The immutable-audit vs right-to-erasure conflict

This is a direct, unavoidable conflict between two hard requirements in the brief, and auditors
will raise it. GDPR Article 17 gives a data subject the right to erasure. This document requires
that audit records are never edited or deleted.

**Resolution:** audit and transfer rows reference accounts by opaque ID only — never by name,
email, or any other personal identifier. Erasure tombstones the `accounts` row, destroying the
personal data, while the audit rows retain the now-meaningless ID and remain intact. The
ownership history stays complete and verifiable; the person is gone from it.

Retention of the transfer record itself rests on legal obligation and legitimate interest — it
is the record of title to a physical asset, comparable to a transaction record — which is a
defensible basis, but it must be **written into the privacy policy before launch**, not argued
for the first time during the audit.

*Trade-off:* audit logs become harder to read for humans, since resolving an actor requires a
join that may return a tombstone. Correct outcome; add an admin tool that renders them.

---

## 11. Deployment and operations

```
CDN / edge ── rate limiting, shape rejection
  └── Resolver           stateless, autoscaled, replica reads only
        ├── Redis        resolution cache, positive and negative
        └── Postgres     read replicas

Application API          authenticated, per-account rate limits
Admin API                network-restricted, step-up auth
Workers                  queue consumers, no ingress
PgBouncer                connection pooling — mandatory at this instance count
Postgres primary         writes; streaming replicas for reads
Object storage           profile media, signed URLs only
Stripe                   webhooks in; never called on a read path
HSM / KMS                NTAG424 master key, never in application memory
```

**Connection pooling is not optional.** Autoscaled stateless services multiply connections
without bound; Postgres does not. PgBouncer in transaction mode sits in front of everything.

**Migrations use expand/contract.** At 12M card rows and 1.2B event rows there is no maintenance
window in which a blocking `ALTER` is acceptable. Add, backfill, dual-write, switch, remove — every
schema change ships as a sequence of independently deployable steps.

**Backups are not a control until a restore has been tested.** Point-in-time recovery, and a
scheduled restore rehearsal with a recorded RTO. "We have backups" is the answer that fails an
audit; "we restored to a specific timestamp last month and it took 41 minutes" is the one that
passes.

### Data residency

At 10M users spanning the EU, US and India, GDPR and India's DPDP Act may impose residency
requirements — and the analytics tier is where the personal data concentrates. The design keeps
this open: `tap_events` is partitioned and regionally separable, and the resolver is stateless
and regionally deployable.

*Recommendation:* confirm target markets before Step 2. Retrofitting residency into a single
global event store is one of the most expensive migrations in this class of system, and the
decision changes the partitioning key.

---

## 12. Threat model summary

| Threat | Control | Where |
| --- | --- | --- |
| Chip rewrite → phishing | Mandatory permanent lock at encoding | §7.3 |
| Chip cloning | NTAG424 DNA, per-card diversified keys | §7.1–7.2 |
| Fleet key compromise | Master key in HSM, never in app or DB | §7.2 |
| Token enumeration | 128-bit CSPRNG, shape validation, negative cache | §3, §5.3 |
| Resolver amplification DoS | Negative caching, miss-rate limiting, single-flight | §5.3–5.4 |
| Unsold card claimed | Scratch code, Argon2id, rate limit, uniform responses | §6 |
| Card-existence oracle | Identical response and timing on claim failure | §6 |
| Ownership hijack | Admin-only transfer, append-only, DB-enforced FKs | §6, §10 |
| Employer/employee URL dispute | Org profiles separate from personal | §2.6 |
| Privilege escalation | Platform and org role scopes never conflated | §9 |
| Session theft | Server-side sessions, rotation with reuse detection | §9 |
| Cancelled card still serving | `serving_state` + drift reconciliation | §2.2 |
| Silent audit divergence | Same-transaction writes, `REVOKE UPDATE, DELETE` | §10 |
| Visitor data over-collection | No raw IP, device class only, partition drops | §8 |
| Erasure vs immutable audit | Opaque actor IDs, account tombstoning | §10 |

---

## 13. Open items requiring your input

Resolved by me where the safer answer was clear. These genuinely need you:

1. **Target markets and data residency** (§11). Changes the analytics partitioning key. Cheapest
   to decide now, most expensive item in this list to retrofit.
2. **Four-table analytics split** (§2.3) — if the reason is per-type retention policy rather than
   volume, tell me and I will reinstate it.
3. **Work-profile permanence in product copy** (§2.6). An architectural decision with a customer
   promise attached; it needs a product owner, not an architect.

---

## Step 2 — Database Design will contain

Every table, column, type, index, and cascade rule. The constraints in §6 written as DDL.
Partitioning DDL and the retention job for `tap_events`. The append-only grants from §10. The
tombstoning mechanism from §10. Migration ordering under expand/contract.
