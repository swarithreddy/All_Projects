const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const box = 20;
const canvasSize = 20;
let snake = [{ x: 9 * box, y: 10 * box }];
let direction = null;
let food = spawnFood();
let score = 0;
let highScore = 0;

document.addEventListener('keydown', changeDirection);
document.getElementById('retryButton').addEventListener('click', retryGame);

function spawnFood() {
  return {
    x: Math.floor(Math.random() * canvasSize) * box,
    y: Math.floor(Math.random() * canvasSize) * box
  };
}

function changeDirection(event) {
  if (event.keyCode === 37 && direction !== 'RIGHT') direction = 'LEFT';
  else if (event.keyCode === 38 && direction !== 'DOWN') direction = 'UP';
  else if (event.keyCode === 39 && direction !== 'LEFT') direction = 'RIGHT';
  else if (event.keyCode === 40 && direction !== 'UP') direction = 'DOWN';
}

function retryGame() {
  document.getElementById('gameOverPrompt').classList.add('hidden');
  snake = [{ x: 9 * box, y: 10 * box }];
  direction = null;
  food = spawnFood();
  score = 0;
  game = setInterval(draw, 100);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < snake.length; i++) {
    ctx.fillStyle = (i === 0) ? 'green' : 'white';
    ctx.fillRect(snake[i].x, snake[i].y, box, box);
    ctx.strokeStyle = 'red';
    ctx.strokeRect(snake[i].x, snake[i].y, box, box);
  }

  ctx.fillStyle = 'red';
  ctx.fillRect(food.x, food.y, box, box);

  let snakeX = snake[0].x;
  let snakeY = snake[0].y;

  if (direction === 'LEFT') snakeX -= box;
  if (direction === 'UP') snakeY -= box;
  if (direction === 'RIGHT') snakeX += box;
  if (direction === 'DOWN') snakeY += box;

  if (snakeX === food.x && snakeY === food.y) {
    food = spawnFood();
    score += 10;
  } else {
    snake.pop();
  }

  let newHead = { x: snakeX, y: snakeY };

  if (snakeX < 0 || snakeX >= canvas.width || snakeY < 0 || snakeY >= canvas.height || collision(newHead, snake)) {
    clearInterval(game);
    if (score > highScore) {
      highScore = score;
    }
    document.getElementById('score').textContent = score;
    document.getElementById('highScore').textContent = highScore;
    document.getElementById('gameOverPrompt').classList.remove('hidden');
  }

  snake.unshift(newHead);
}

function collision(head, array) {
  for (let i = 0; i < array.length; i++) {
    if (head.x === array[i].x && head.y === array[i].y) {
      return true;
    }
  }
  return false;
}

let game = setInterval(draw, 100);
