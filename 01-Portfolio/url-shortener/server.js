const express = require("express");
const connectDB = require("./config/db");
const urlRoutes = require("./routes/urlRoutes");
const limiter = require("./middleware/rateLimiter");

require("./jobs/cleanupExpired");

const app = express();

app.use(express.json());
app.use(limiter);

// serve frontend
app.use(express.static("public"));

connectDB();

app.use("/", urlRoutes);

app.listen(3000, () => {
  console.log("Server started on port 3000");
});