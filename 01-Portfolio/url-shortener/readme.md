# URL Shortener — Step-by-Step Explanation (Interview-Ready)

This document explains your project from first principles to production features in simple language you can revise quickly before interviews.

---

## 1) What is a URL Shortener?

A URL shortener converts a **long link** like:

```
https://www.google.com/search?q=very+long+query...
```

into a **small link** like:

```
http://localhost:3000/Ab3kP
```

When someone opens the short link, they are **redirected** to the original long link.

Famous examples: TinyURL, Bitly.

---

## 2) What Problems Are We Solving?

| Problem                     | Our Solution                       |
| --------------------------- | ---------------------------------- |
| Long URLs are hard to share | Generate short codes using Base62  |
| Two users may get same code | Collision handling + DB uniqueness |
| Links may expire            | Expiry date stored in DB           |
| Need usage data             | Click counter (analytics)          |
| Bots may spam               | Rate limiting                      |
| DB may fill with old links  | Cron cleanup job                   |
| Users want custom names     | Custom alias support               |
| Invalid links               | URL validation                     |

---

## 3) Tech Stack

* Backend: **Express.js**
* Database: **MongoDB**
* Scheduler: **node-cron**
* Security: **express-rate-limit**
* Frontend: HTML + Fetch API

---

## 4) High-Level Working (Big Picture)

```
User → POST /shorten → Save in DB → Get short URL
User → GET /:code → Read DB → Increase clicks → Redirect
```

---

## 5) Database Design (Very Important for Interviews)

Collection: `urls`

| Field     | Type            | Purpose               |
| --------- | --------------- | --------------------- |
| shortCode | String (unique) | The small code in URL |
| longUrl   | String          | Original link         |
| clicks    | Number          | Analytics counter     |
| expiry    | Date            | Auto invalidation     |
| createdAt | Date            | Timestamp             |

Why unique index on `shortCode`?
→ Prevent duplicates at database level.

---

## 6) Base62 Encoding (How short codes are made)

We convert numbers to characters:

```
a-z → 26
A-Z → 26
0-9 → 10
Total = 62 characters
```

Large numbers become small strings.

Example:

```
999999 → "g7K"
```

This is why URLs are short.

---

## 7) API Endpoints

### POST `/shorten`

Input:

```json
{
  "longUrl": "...",
  "customCode": "...",
  "expiryDays": 2
}
```

Flow:

1. Validate URL
2. If custom alias → check availability
3. Else → generate Base62 code
4. Handle collision (retry)
5. Save to DB
6. Return short URL

---

### GET `/:code` (Redirect)

Flow:

1. Find code in DB
2. If not found → 404
3. If expired → 410
4. Increase clicks
5. Redirect to long URL

---

### GET `/analytics/:code`

Returns:

* longUrl
* clicks
* created time
* expiry

---

## 8) Collision Handling (Real System Concept)

Why collision happens?
Two requests at same millisecond may generate same number.

Solution:

* Unique index in DB
* While loop retry until unique code found

This is **production thinking**.

---

## 9) Duplicate Long URL Detection

If same long URL already exists:
→ Return existing short URL instead of creating new one.

Saves space and keeps system clean.

---

## 10) Custom Alias

User can request:

```
/mycollege
```

We check if already taken.
If free → assign.

---

## 11) Rate Limiting (Security)

Using **express-rate-limit**:

* 20 requests per 15 minutes per IP
* Prevents bots/spam attacks

---

## 12) Cleanup Job (Automation)

Using **node-cron**:

Every hour:

* Delete expired links from **MongoDB**

Keeps DB optimized.

---

## 13) Frontend

Simple HTML page:

* Enter URL
* Optional alias
* Calls API using `fetch`
* Shows short link

---

## 14) Complete Flow (End to End)

```
Browser UI
   ↓
Express Route
   ↓
Validation
   ↓
Generate Code
   ↓
MongoDB Save
   ↓
Return Short URL
   ↓
User Opens Short URL
   ↓
DB Lookup → Click++ → Redirect
```

---

## 15) Interview Questions You Can Now Answer

**Q: How do you prevent short code collision?**
DB unique index + retry generation loop.

**Q: How do you scale this system?**

* Move to Redis cache for reads
* Use load balancer
* Shard database
* Pre-generate codes

**Q: How do you handle expired links?**
Expiry field + cron cleanup + runtime check.

**Q: How do you stop abuse?**
Rate limiter middleware.

**Q: Why Base62?**
Max characters in minimum length, URL-safe.

**Q: How is analytics tracked?**
Increment clicks on each redirect.

---

## 16) Folder Structure (Mental Map)

```
config/db.js
models/urlModel.js
routes/urlRoutes.js
utils/base62.js
utils/generateShortCode.js
utils/validateUrl.js
middleware/rateLimiter.js
jobs/cleanupExpired.js
public/index.html
server.js
```

---

## 17) What Makes This “Production-Grade”

You didn’t just shorten URLs. You added:

* Validation
* Security
* Collision safety
* Analytics
* Expiry
* Cleanup automation
* Custom aliases
* Frontend

That’s **system design thinking**, not just coding.

---

## 18) One-Line Summary (for interviews)

> “I built a full-stack URL shortener using Express and MongoDB with Base62 encoding, collision handling, analytics, custom aliases, rate limiting, and automated expiry cleanup to simulate a production-ready TinyURL/Bitly-like system.”
