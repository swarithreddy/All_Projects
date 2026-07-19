const cron = require("node-cron");
const Url = require("../models/urlModel");

// Runs every 1 hour
cron.schedule("0 * * * *", async () => {
  try {
    const now = new Date();

    const result = await Url.deleteMany({
      expiry: { $ne: null, $lt: now }
    });

    console.log(`Cleanup Job: Deleted ${result.deletedCount} expired URLs`);
  } catch (err) {
    console.error("Cleanup Job Error:", err.message);
  }
});