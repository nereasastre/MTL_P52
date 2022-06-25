// function (ctx) {
//   let index = ctx.dataIndex;
  
//   let logRMS = RMS_ARRAY[index];
//   if (logRMS < 0) logRMS = 0;
  
//   // console.info("borderWidth: ", logRMS);
//   return logRMS * 8; // max 6 pixels wide
// }
const DATA = {
  "labels": [],
  "yLabels": [],
  "datasets": [{
    "data": [],
    "fill": false,
    "type": "line",
    "pointRadius": 0,
    "borderWidth": 5,
    "fontColor": "#555",
    "yAxisID": "main-y-axis",
    "backgroundColor": "",
    "label": "",
    "borderColor": function(ctx) {
      return setColorGradient(ctx);
    },
    "pointBorderColor": "transparent"
  }, {
    "data": [],
    "fill": false,
    "type": "line",
    "pointRadius": 0,
    "borderWidth": 1,
    "label": "background-fill-bottom",
    "yAxisID": "background-area-y-axis",
    "borderColor": "rgba(224, 224, 224, 0.5)",
    "backgroundColor": "rgba(224, 224, 224, 0.5)",
    "pointBorderColor": "transparent"
  }, {
    "data": [],
    "fill": "-1",
    "type": "line",
    "pointRadius": 0,
    "borderWidth": 1,
    "label": "background-fill-top",
    "yAxisID": "background-area-y-axis",
    "borderColor": "rgba(224, 224, 224, 0.5)",
    "backgroundColor": "rgba(224, 238, 242, 0.5)",
    "pointBorderColor": "transparent"
  }]
};

const OPTIONS = {
  "responsive": true,
  "animation": false,
  "title": {
    "display": false
  },
  "maintainAspectRatio": true,
  "legend": {
    "labels": {
      "fontSize": 11,
      "boxWidth": 22,
      "fontColor": "rgb(50,50,50)",
      "filter": function (item, chart) {
        return item.text.includes("Pitch");
      }
    }
  },
  "layout": {
    "padding": {
      "top": 0,
      "left": 10,
      "right": 10,
      "bottom": 0
    }
  },
  "scales": {
    "xAxes": [{
      "barPercentage": 1.00,
      "categoryPercentage": 1.00,
      "gridLines": {
        "display": true
      },
      "ticks": {
        "fontSize": 13,
        "fontColor": "rgb(50,50,50)",
        "maxRotation": 15,
        "autoSkip": true,
        "autoSkipPadding": 10
      },
      "scaleLabel": {
        "display": true,
        "fontSize": 16,
        "fontColor": "rgb(50,50,50)",
        "labelString": "Time (seconds)"
      }
    }],
    "yAxes": [{
        "id": "main-y-axis",
        "type": "logarithmic",
        "position": "left",
        "stacked": true,
        "gridLines": {
          "display": false
        },
        "ticks": {
          "min": 0,
          "max": 4000,
          "fontSize": 13,
          "fontColor": "rgb(50,50,50)",
          "stepSize": 0,
          "maxTicksLimit": 100,
          "callback": function(...args) {
            const value = Chart.Ticks.formatters.logarithmic.call(this, ...args);
            if (value.length) {
              return Number(value).toLocaleString()
            }
            return value;
          }
        },
        "scaleLabel": {
          "display": true,
          "fontSize": 16,
          "fontColor": "rgb(50,50,50)",
          "labelString": "Pitch (Hz)"
        }
      },
      {
        "id": "pitchclass-y-axis",
        "position": "right",
        "type": "category",
        "stacked": false,
        "gridLines": {
          "display": true,
          "color": [],
          "z": 5,
          "drawTicks": true
        },
        "ticks": {
          "fontSize": 13,
          "fontColor": "rgb(50,50,50)",
          "maxTicksLimit": 100,
          "callback": function(value, index, values) {
            return value;
          }
        },
        "scaleLabel": {
          "display": true,
          "fontSize": 16,
          "fontColor": "rgb(50,50,50)",
          "labelString": "Note"
        }
      },
      /* auxiliary axes */
      {
        "id": "background-area-y-axis",
        "position": "left",
        "type": "logarithmic",
        "stacked": true,
        "gridLines": {
          "display": false
        },
        "ticks": {
          "min": 0,
          "max": 4000,
          "fontSize": 11,
          "fontColor": "rgb(50,50,50)",
          "callback": function(...args) {
            const value = Chart.Ticks.formatters.logarithmic.call(this, ...args);
            if (value.length) {
              return Number(value).toLocaleString()
            }
            return value;
          }
        },
        "display": false
      }
    ]
  }
};

const PITCH_CLASS_COLORS = {
  'C': 'rgb(0, 100, 0)', 
  'C#': 'rgb(130, 130, 130)', 
  'D': 'rgb(0, 100, 0)', 
  'D#': 'rgb(130, 130, 130)', 
  'E': 'rgb(0, 100, 0)', 
  'F': 'rgb(0, 100, 0)', 
  'F#': 'rgb(130, 130, 130)', 
  'G': 'rgb(0, 100, 0)', 
  'G#': 'rgb(130, 130, 130)', 
  'A': 'rgb(0, 100, 0)', 
  'A#': 'rgb(130, 130, 130)', 
  'B': 'rgb(0, 100, 0)'
};

// const PITCH_CLASS_COLORS = {
//   'C': 'hsl(210, 25%, 50%)', 
//   'C#': 'hsl(240, 25%, 50%)', 
//   'D': 'hsl(270, 25%, 50%)', 
//   'D#': 'hsl(300, 25%, 50%)', 
//   'E': 'hsl(330, 25%, 50%)', 
//   'F': 'hsl(0, 25%, 50%)', 
//   'F#': 'hsl(30, 25%, 50%)', 
//   'G': 'hsl(60, 25%, 50%)', 
//   'G#': 'hsl(90, 25%, 50%)', 
//   'A': 'hsl(120, 25%, 50%)', 
//   'A#': 'hsl(150, 25%, 50%)', 
//   'B': 'hsl(180, 25%, 50%)'
// };

const INSTRUMENT_MINS = {
  'Guitar': 82.407, 
  'Violin': 196, 
  'Flute': 261.63, 
  'Bass': 41.2, 
  'Cello': 65.41, 
  'Double Bass': 41.2,
  'Sax': 55, 
  'Trombone': 65.41, 
  'Bass Horn': 43.65, 
  'Ukulele': 261.63, 
  'Voice': 73.42, 
  'Default': 41.2
}
const INSTRUMENT_MAXS = {
  'Guitar': 622.255, 
  'Violin': 2589.07, 
  'Flute': 2637.02, 
  'Bass': 207.65, 
  'Cello': 622.254, 
  'Double Bass': 207.65, 
  'Sax': 987.77, 
  'Trombone': 659.25, 
  'Bass Horn': 392, 
  'Ukulele': 880, 
  'Voice': 1046.56, 
  'Default': 2589.07
}

const SEMITONE_RATIO = Math.pow(2, 1/12);
const CONFIDENCE_ARRAY = Array(30).fill(0);
const RMS_ARRAY = Array(30).fill(0);
let rms_pointer = RMS_ARRAY;

function getPitchScale(instrument) {
  let freq = INSTRUMENT_MINS[instrument] / SEMITONE_RATIO; // fill from minimum instrument frequency - a semitone
  let semitoneScale = [];
  
  do {
      semitoneScale.push(freq);
      freq = freq * SEMITONE_RATIO;
  } while (freq <= INSTRUMENT_MAXS[instrument] * SEMITONE_RATIO) // fill up to maximum instrument frequency + semitone
  return semitoneScale;
} 

function freqToPitchClass(f) {
  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const C1 = 440 * Math.pow(SEMITONE_RATIO, -45);
  const semitonesAboveC1 = Math.round(12 * Math.log2(f / C1));
  return keys[semitonesAboveC1 % 12];
}

function setColorGradient(ctx) {
  // "#ff5144": hue = 4
  const chartArea = ctx.chart.chartArea;
  if (!chartArea) return null;

  const canvasCtx = ctx.chart.ctx;
  let gradient = canvasCtx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
  const transparency = rms_pointer;

  const dataSize = ctx.dataset.data.length;
  const stops = Array.from(Array(dataSize).keys()).map(i => i/dataSize); // normalized (0 - 1) gradient stops positions
  stops.forEach((s, idx) => {
    gradient.addColorStop(s, `rgba(255, 81, 69, ${transparency[idx]})`)
  });
  
  return gradient;
}

function setPitchAxis(){
  AXES_PITCHES = getPitchScale(instrument);
  // fill in data, labels, and setup axes

  for (var i = 0; i < 30; i++) DATA.datasets[0].data.push(100);
  for (var i = 0; i < 30; i++) DATA.datasets[1].data.push(AXES_PITCHES[0]);
  for (var i = 0; i < 30; i++) DATA.datasets[2].data.push(AXES_PITCHES.slice(-1)[0]);

  DATA.yLabels = [];
  for (var p of AXES_PITCHES) {
      DATA.yLabels.push(freqToPitchClass(p));
  }
  DATA.yLabels.reverse();

  OPTIONS.scales.yAxes[0].ticks.min = AXES_PITCHES[0];
  OPTIONS.scales.yAxes[0].ticks.max = AXES_PITCHES.slice(-1)[0];
  OPTIONS.scales.yAxes[2].ticks.min = AXES_PITCHES[0];
  OPTIONS.scales.yAxes[2].ticks.max = AXES_PITCHES.slice(-1)[0];

  OPTIONS.scales.yAxes[1].gridLines.color = [];
  for (var n of DATA.yLabels) {
    OPTIONS.scales.yAxes[1].gridLines.color.push(PITCH_CLASS_COLORS[n]);
  }
}

instrument = 'Default';
setPitchAxis();

function changeInstrument(){
  instrument = document.getElementById("instrument").value;
  setPitchAxis();
}
