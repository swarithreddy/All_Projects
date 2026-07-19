const encode = require("./base62");

function generate() {
  const randomNum = Date.now() + Math.floor(Math.random() * 10000);
  return encode(randomNum);
}

module.exports = generate;