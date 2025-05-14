
const drawPixel = document.getElementById('drawpixel')
const drawpad = document.getElementById('drawpad')
const responseElem = document.getElementById('response')


for (let i = 0; i < 28 * 28; i++) {
  const pixel = drawPixel.cloneNode();
  pixel.style.backgroundColor = 'rgb(0,0,0)'
  pixel.curShade = 0
  pixel.style.visibility = 'visible'

  pixel.addEventListener('mousemove', (event) => {

    target = event.currentTarget

    if (event.buttons === 1) {
      target.curShade += 170
      target.curShade = Math.min(255, target.curShade)
      pixel.style.backgroundColor = `rgb(${target.curShade}, ${target.curShade}, ${target.curShade})`

    }

  })

  drawpad.appendChild(pixel);
}

function getAllPixels() {
  let pixelsArray = [];

  const children = drawpad.children
  for (let i = 0; i < 28; i++) {
    let curArray = [];
    for (let j = 0; j < 28; j++) {
      const pixel = children[i * 28 + j];
      curArray.push(pixel.curShade);
    }
    pixelsArray.push(curArray);
  }

  return pixelsArray;
}

function clearDrawing() {
  const children = drawpad.children
  for (let i = 0; i < 28 * 28; i++) {
    const pixel = children[i]
    pixel.curShade = 0
    pixel.style.backgroundColor = 'rgb(0,0,0)'
  }
}

function convertArr(acc, s, i) {
  return acc + i + ": " + s + '\n'
}

function predictDrawing() {
  $.ajax({
    type: "POST",
    url: "http://127.0.0.1:5000/ai",
    data: JSON.stringify({ "pixels": getAllPixels() }),
    contentType: "application/json",
    success: function (resp) {
      responseElem.innerText = resp["response"].reduce(convertArr, "") + `I would predict a ${resp["maxNo"]}`
    }
  })

}