const drawPixel = document.getElementById('drawpixel');
const drawpad = document.getElementById('drawpad');
const responseElem = document.getElementById('response');

const width = 28;
const height = 28;
const pixels = []; // <-- store all pixels for easy access

// Initialize all pixels
for (let i = 0; i < width * height; i++) {
  const pixel = drawPixel.cloneNode();
  pixel.style.backgroundColor = 'rgb(0,0,0)';
  pixel.curShade = 0;
  pixel.style.visibility = 'visible';

  pixels.push(pixel); // <-- add to flat pixel array
  drawpad.appendChild(pixel);
}

// helper to safely update a pixel’s shade by coordinates
function updateShade(x, y, delta) {
  if (x < 0 || y < 0 || x >= width || y >= height) return;
  const index = y * width + x;
  const pixel = pixels[index];
  pixel.curShade = Math.min(255, pixel.curShade + delta);
  pixel.style.backgroundColor = `rgb(${pixel.curShade}, ${pixel.curShade}, ${pixel.curShade})`;
}

// Add brush behavior to all pixels
pixels.forEach((pixel, idx) => {
  pixel.addEventListener('mousemove', (event) => {
    if (event.buttons !== 1) return;

    const x = idx % width;
    const y = Math.floor(idx / width);

    // 3x3 circle brush mask — center stronger, edges lighter
    const brush = [
      { dx:  0, dy:  0, weight: 170 },
      { dx: -1, dy:  0, weight: 60 },
      { dx:  1, dy:  0, weight: 60 },
      { dx:  0, dy: -1, weight: 60 },
      { dx:  0, dy:  1, weight: 60 },
      { dx: -1, dy: -1, weight: 20 },
      { dx: -1, dy:  1, weight: 20 },
      { dx:  1, dy: -1, weight: 20 },
      { dx:  1, dy:  1, weight: 20 },
    ];

    for (const { dx, dy, weight } of brush) {
      updateShade(x + dx, y + dy, weight);
    }
  });
});

// Get current pixel shades as 2D array
function getAllPixels() {
  let pixelsArray = [];
  for (let i = 0; i < height; i++) {
    let curArray = [];
    for (let j = 0; j < width; j++) {
      const pixel = pixels[i * width + j]; // <-- use new pixel array
      curArray.push(pixel.curShade);
    }
    pixelsArray.push(curArray);
  }
  return pixelsArray;
}

// Clear drawing board
function clearDrawing() {
  for (let i = 0; i < width * height; i++) {
    const pixel = pixels[i]; // <-- use new pixel array
    pixel.curShade = 0;
    pixel.style.backgroundColor = 'rgb(0,0,0)';
  }
}

// Convert predictions to string
function convertArr(acc, s, i) {
  return acc + i + ": " + s + '\n';
}

// Send drawing to backend for prediction
function predictDrawing() {
  $.ajax({
    type: "POST",
    url: "http://127.0.0.1:5000/ai",
    data: JSON.stringify({ "pixels": getAllPixels() }),
    contentType: "application/json",
    success: function (resp) {
      responseElem.innerText =
        resp["response"].reduce(convertArr, "") +
        `I would predict a ${resp["maxNo"]}`;
    }
  });
}
