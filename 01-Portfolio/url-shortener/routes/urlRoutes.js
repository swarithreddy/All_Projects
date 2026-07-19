const express = require("express");
const router = express.Router();

const Url = require("../models/urlModel");
const generateShortCode = require("../utils/generateShortCode");
const isValidUrl = require("../utils/validateUrl");

// CREATE SHORT URL (with custom alias)
router.post("/shorten", async (req, res) => {
  try {
    const { longUrl, expiryDays, customCode } = req.body;

    // 1) Validate URL
    if (!isValidUrl(longUrl)) {
      return res.status(400).json({ error: "Invalid URL format" });
    }

    // 2) If custom alias is provided
    let shortCode;

    if (customCode) {
      const existingAlias = await Url.findOne({ shortCode: customCode });
      if (existingAlias) {
        return res.status(400).json({ error: "Custom alias already taken" });
      }
      shortCode = customCode;
    } else {
      // 3) Check if URL already shortened
      const existing = await Url.findOne({ longUrl });
      if (existing) {
        return res.json({
          shortUrl: `http://localhost:3000/${existing.shortCode}`,
          message: "URL already shortened"
        });
      }

      // 4) Generate unique code
      let exists = true;
      while (exists) {
        shortCode = generateShortCode();
        const check = await Url.findOne({ shortCode });
        if (!check) exists = false;
      }
    }

    // 5) Expiry
    let expiry = null;
    if (expiryDays) {
      expiry = new Date();
      expiry.setDate(expiry.getDate() + expiryDays);
    }

    // 6) Save
    await Url.create({
      shortCode,
      longUrl,
      expiry
    });

    res.json({
      shortUrl: `http://localhost:3000/${shortCode}`
    });

  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});


// REDIRECT
router.get("/:code", async (req, res) => {
  try {
    const { code } = req.params;

    const url = await Url.findOne({ shortCode: code });

    if (!url) return res.status(404).json({ error: "URL not found" });

    if (url.expiry && new Date() > url.expiry) {
      return res.status(410).json({ error: "Link expired" });
    }

    url.clicks += 1;
    await url.save();

    res.redirect(url.longUrl);

  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});


// ANALYTICS
router.get("/analytics/:code", async (req, res) => {
  try {
    const { code } = req.params;

    const url = await Url.findOne({ shortCode: code });

    if (!url) return res.status(404).json({ error: "URL not found" });

    res.json(url);

  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

module.exports = router;