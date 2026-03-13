export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CSV upload (store in KV)
    if (request.method === 'POST' && url.pathname === '/api/update') {
      const contentType = request.headers.get('Content-Type');
      if (!contentType || !contentType.includes('text/csv')) {
        return new Response('Invalid content type. Expected text/csv.', { status: 400 });
      }
      const csvData = await request.text();
      await env.TEMP_KV.put('latest_csv', csvData);
      const uploadTime =
        request.headers.get('X-Upload-Timestamp') ||
        new Date().toISOString();
      await env.TEMP_KV.put('latest_csv_time', uploadTime);
      return new Response('CSV uploaded successfully with timestamp stored.', { status: 200 });
    }

    // Serve index.html
    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(
        `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rubin Summit Temperature Forecast</title>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/highcharts.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/highcharts-more.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/modules/exporting.js"></script>
  <style>
.twilight-flex {
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: flex-start;
  gap: 2.4rem;
  margin: 2em auto 0 auto;
  width: 94vw;
  max-width: 1250px;
}
.twilight-box {
  flex: 1 1 0;
  max-width: 420px;
  min-width: 190px;
  background: #b22222;      /* firebrick */
  border-radius: 18px;
  box-shadow: 0 6px 32px 0 #0002;
  padding: 1.25em 0.5em 1.1em 0.5em;
  text-align: center;
  color: #fff;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}
.twilight-timebox {
  max-width: 420px;
  min-width: 190px;
  background: #555c65;
  border-radius: 18px;
  box-shadow: 0 2px 16px 0 #0002;
  padding: 1.05em 0.5em 1.0em 0.5em;
  text-align: center;
  color: #fff;
  margin-top: 1.1em;
  margin-bottom: 0.5em;
}
.twilight-forecast-uncertainty {
  display: block;
  font-size: 1.1rem;
  color: #f5e9e9;
  margin-top: 0.35em;
  letter-spacing: 0.01em;
  opacity: 0.93;
}
.twilight-time, .twilight-remaining {
  color: #fff;
}
.twilight-label {
  color: #ccc;
  font-size: 1.19rem;
  margin-bottom: 0.19em;
  letter-spacing: 0.01em;
}
.twilight-temp {
  font-size: 3.3rem;
  font-family: 'SF Mono', 'Menlo', 'Consolas', 'monospace';
  font-weight: 600;
  color: #fff;
  line-height: 1.08;
  margin-bottom: 0.18em;
}
.twilight-time {
  font-size: 1.8rem;
  font-family: 'SF Mono', 'Menlo', 'Consolas', 'monospace';
  font-weight: 600;
  color: #fff;
  line-height: 1.18;
  margin-bottom: 0.12em;
}
.twilight-remaining {
  font-size: 1.31rem;
  color: #f0f0f0;
  margin-top: 0.22em;
  opacity: 0.95;
  letter-spacing: 0.02em;
}
.deg {
  font-size: 1.2rem;
  vertical-align: super;
  margin-left: 2px;
  color: #ccc;
}
.twilight-meta {
  font-size: 1.07rem;
  color: #f0f0f0;
  margin-top: 0.22em;
  opacity: 0.95;
}
#container {
  flex: 2 1 0;
  min-width: 350px;
  min-height: 470px;
  height: 510px;
  margin: 0;
  background: #18181a;
  border-radius: 16px;
  box-shadow: 0 3px 18px 0 #0002;
}
@media (max-width: 1000px) {
  .twilight-flex {
    flex-direction: column;
    gap: 1.3rem;
    width: 99vw;
    align-items: center;          /* NEW: centre children horizontally */
  }
  #container {
    width: 99vw;
    min-width: 200px;
    height: 410px;
    border-radius: 20px;
    box-shadow: 0 3px 18px 0 #0002;
    padding: 14px;          /* NEW: extra inner space */
  }
}
@keyframes flipY {
  0%   { transform: rotateX(0deg); }
  50%  { transform: rotateX(-90deg); }
  100% { transform: rotateX(0deg); }
}
.flip-animate {
  display: inline-block;
  animation: flipY 0.5s ease-in-out;
  backface-visibility: hidden;
}
</style>
</head>
<body>
  <h1 style="text-align:center;">Rubin Summit Temperature Forecast</h1>
  <h2 style="text-align:center;">Forecast of the Day</h2>
  <div id="lastUpdate" style="text-align:center;font-size:0.9rem;color:#555;margin-top:0.25rem;"></div>

  <div class="twilight-flex">
    <div>
      <div class="twilight-box">
        <div class="twilight-label">Forecast at Twilight</div>
        <div class="twilight-temp" id="twilight-forecast">--<span class="deg">°C</span></div>
        <span class="twilight-forecast-uncertainty" id="twilight-forecast-uncertainty">± -- °C</span>
        <div class="twilight-meta" id="twilight-actual">Actual: -- °C (Weather Tower)</div>
      </div>
      <div class="twilight-timebox">
        <div class="twilight-label">Current Time</div>
        <div class="twilight-time" id="current-time">--:-- CLT</div>
        <div class="twilight-label" style="margin-top:0.6em;">Twilight Time</div>
        <div class="twilight-time" id="twilight-time">--:-- CLT</div>
        <div class="twilight-remaining" id="twilight-remaining">--h --min</div>
      </div>
    </div>
    <div id="container"></div>
  </div>
  <p style="text-align:center;font-size:0.85rem;color:#555;margin-top:0.8rem;">
    <em>Note: the forecast model is continuously updated throughout the day.</em>
  </p>

  <script>
      const statusDiv = document.createElement('div');   // spot for user messages
      statusDiv.style.cssText = 'text-align:center;color:red;margin-top:1rem;';
      document.body.prepend(statusDiv);
      const lastUpdateDiv = document.getElementById('lastUpdate');

      // ---------- live‑update helpers ----------
      function updateBadge(latestTs) {
        function render() {
          // Show delay in whole minutes, always non‑negative
          const mins = Math.floor(Math.abs(Date.now() - latestTs) / 60000);
          lastUpdateDiv.textContent = 'Last update: ' + mins + ' min ago';
          lastUpdateDiv.style.color = mins > 30 ? 'red' : '#555';
        }
        render();
        if (updateBadge.timer) clearInterval(updateBadge.timer);
        updateBadge.timer = setInterval(render, 60 * 1000);
      }

      // ---- Show current time in CLT ----
      function updateCurrentTimeCL() {
        const nowCL = new Date().toLocaleTimeString('en-US', {
          timeZone: 'America/Santiago',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        });
        const ct = document.getElementById('current-time');
        if (ct && ct.textContent !== nowCL + ' CLT') {
          ct.classList.add('flip-animate');
          ct.textContent = nowCL + ' CLT';
          ct.addEventListener('animationend', () => ct.classList.remove('flip-animate'), { once: true });
        }
      }
      updateCurrentTimeCL();
      setInterval(updateCurrentTimeCL, 60 * 1000);

      function fetchAndRedraw() {
        fetch('/api/forecast')
          .then(r => {
            if (!r.ok) throw new Error('Server replied ' + r.status + ' ' + r.statusText);
            return r.text();
          })
          .then(csv => {
            if (!csv.trim()) throw new Error('Empty CSV returned');

            // ---- parse CSV (original logic retained) ----
            const lines  = csv.trim().split(String.fromCharCode(10));
            const header = lines.shift().split(',');
            const idx = n => header.indexOf(n);
            const safe = v => { const n = parseFloat(v); return isNaN(n) ? null : +n.toFixed(1); };

            let latestPastTs = null;

            const SOURCE_COLORS = {
              prophet: { line: 'firebrick', band: 'rgba(178,34,34,0.25)' },
              nbeats:  { line: 'steelblue', band: 'rgba(70,130,180,0.25)' }
            };
            const DEFAULT_COLOR = SOURCE_COLORS.prophet;

            const tempMin = [], tempActual = [], tempMax = [];
            const fcMin = [], forecast = [], fcMax = [];
            const sunset = [];
            const sunrise = [];
            let forecastSource = null;

            lines.forEach(l => {
              if (!l.trim()) return;
              const c = l.split(',');
              const ts = new Date(c[idx('timestamp')]).getTime();
              if (ts <= Date.now()) latestPastTs = ts;
              if (c[idx('sunset')].toLowerCase() === 'true') sunset.push(ts);
              if (c[idx('sunrise')] && c[idx('sunrise')].toLowerCase() === 'true') sunrise.push(ts);
              tempActual.push([ts, safe(c[idx('temp_actual')])]);
              forecast.push([ts, safe(c[idx('forecast')])]);
              tempMin.push([ts, safe(c[idx('temp_min')])]);
              tempMax.push([ts, safe(c[idx('temp_max')])]);
              fcMin.push([ts, safe(c[idx('forecast_min')])]);
              fcMax.push([ts, safe(c[idx('forecast_max')])]);
              if (!forecastSource && idx('forecast_source') >= 0) {
                const src = (c[idx('forecast_source')] || '').trim().toLowerCase();
                if (src) forecastSource = src;
              }
            });

            const palette = SOURCE_COLORS[forecastSource] || DEFAULT_COLOR;
            const sourceLabel = forecastSource
              ? forecastSource.charAt(0).toUpperCase() + forecastSource.slice(1)
              : 'Forecast';
            
            // badge
            const badgeTs = latestPastTs ?? forecast.at(0)[0];
            updateBadge(badgeTs);

            const obsBand  = tempMin.map((d, i) => [d[0], d[1], tempMax[i][1]]);
            const predBand = fcMin.map((d, i) => [d[0], d[1], fcMax[i][1]]);
            // ---- Compute average night length and build bands ----
            let nightMs = 0;
            // If we have at least one sunrise and one sunset, compute day/night
            if (sunrise.length > 0 && sunset.length > 0) {
              const lastSunrise = sunrise[sunrise.length - 1];
              const lastSunset  = sunset[sunset.length - 1];
              const dayMs = lastSunset - lastSunrise;           // daylight duration
              if (dayMs > 0 && dayMs < 86400000) {
                nightMs = 86400000 - dayMs;
              }
            }
            // Fallback: 12 h if calculation failed
            if (nightMs === 0) nightMs = 12 * 3600 * 1000;       // 12 h in ms

            const nightBands = sunset.map(function (sun) {
              return {
                color: 'rgba(85, 92, 101, 0.1)',
                from: sun,
                to: sun + nightMs,
                label: { text: 'Night Time', style: { color: '#555c65', fontWeight: 600 } },
                zIndex: 0
              };
            });
              
            Highcharts.setOptions({ time: { timezone: 'America/Santiago' } });

            if (window.chart) window.chart.destroy();

            window.chart = Highcharts.chart('container', {
              chart: { type: 'spline', zoomType: 'x', spacing: [40, 40, 20, 20],   // top, right, bottom, left
                       resetZoomButton: { position: { align: 'right', verticalAlign: 'top', x: 0, y: 0 } } },
              title: { text: null },
                xAxis: {
                type: 'datetime',
                title: { text: 'Time (CLT)' },
                plotBands: nightBands,
                plotLines: sunset.map(function(ts) {
                  return {
                    value: ts, color: 'gray', width: 2, dashStyle: 'Dash',
                    label: { text: 'Sunset', rotation: 90, textAlign: 'left', style: { color: 'gray' } },
                    zIndex: 5
                  };
                })
              },
              yAxis: { title: { text: 'Temperature (°C)' } },
              tooltip: {
                shared: true,
                xDateFormat: '%Y-%m-%d %H:%M',
                formatter() {
                  let s = '<b>' + Highcharts.dateFormat('%Y-%m-%d %H:%M', this.x) + '</b><br/>';
                  this.points.forEach(p => {
                    const n = p.series.name, col = p.color;
                    if (n === '(max-min)' || n === '68% cfi') {
                      s += '<span style="color:' + col +
                           '">●</span> (max‑min): <b>' + (p.point.high - p.point.low).toFixed(2) + '°C</b><br/>';
                    } else {
                      s += '<span style="color:' + col + '">●</span> ' + n +
                           ': <b>' + p.y.toFixed(2) + '°C</b><br/>';
                    }
                  });
                  return s;
                }
              },
              series: [
                { name: '(max-min)', type: 'arearange', data: obsBand,
                  color: '#8080804d', lineWidth: 0, marker: { enabled: false }, zIndex: 0 },
                { name: 'Weather Tower', data: tempActual, color: 'black', zIndex: 1, connectNulls: false },
                { name: '68% cfi', type: 'arearange', data: predBand,
                  color: palette.band, lineWidth: 0, marker: { enabled: false }, zIndex: 0 },
                { name: sourceLabel + ' Forecast', data: forecast, color: palette.line, zIndex: 1, connectNulls: false }
              ],
              credits: { enabled: false }
            });

            // ----- Get sunset (twilight) time from CSV -----
            var twilightUTC = null;
            if (sunset.length > 0) {
              twilightUTC = sunset[sunset.length - 1];
            } else {
              var now = new Date();
              var dtfCL = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Santiago' });
              var parts = dtfCL.formatToParts(now);
              var todayY = +parts.find(function(x) { return x.type === 'year'; }).value;
              var todayM = +parts.find(function(x) { return x.type === 'month'; }).value - 1;
              var todayD = +parts.find(function(x) { return x.type === 'day'; }).value;
              twilightUTC = santiagoLocalTimeToUTC(todayY, todayM, todayD, 18, 0);
            }
            function closestIdx(arr, target) {
              return arr.reduce(function(bestIdx, pair, i, a) {
                return Math.abs(pair[0] - target) < Math.abs(a[bestIdx][0] - target) ? i : bestIdx;
              }, 0);
            }
            var idxTwilight = closestIdx(forecast, twilightUTC);
            var forecastTwilight = forecast[idxTwilight][1];
            var actualTwilight = tempActual[idxTwilight][1];
            var fMinVal = fcMin[idxTwilight][1];
            var fMaxVal = fcMax[idxTwilight][1];
            var uncTwilight = (fMinVal != null && fMaxVal != null) ? (fMaxVal - fMinVal)/2.0 : NaN;

            // --- Update forecast box ---
            document.getElementById('twilight-forecast').innerHTML =
              (forecastTwilight !== null ? forecastTwilight.toFixed(1) : '--') + '<span class="deg">°C</span>';
            document.getElementById('twilight-forecast-uncertainty').textContent =
              (isFinite(uncTwilight) && uncTwilight > 0)
                ? '\u00B1 ' + uncTwilight.toFixed(1) + '\u00B0C'
                : '\u00B1 -- \u00B0C';
            document.getElementById('twilight-actual').textContent =
              (actualTwilight !== null ? actualTwilight.toFixed(1) : '--') + ' °C (Weather Tower)';

            // --- Update Twilight Time box ---
            try {
              var dtTwilight = new Date(twilightUTC);
              var twilightCLTime = dtTwilight.toLocaleTimeString('en-US', {
                timeZone: 'America/Santiago',
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
              });
              var twElem = document.getElementById('twilight-time');
              if (twElem && twElem.textContent !== twilightCLTime + ' CLT') {
                twElem.classList.add('flip-animate');
                twElem.textContent = twilightCLTime + ' CLT';
                twElem.addEventListener('animationend', () => twElem.classList.remove('flip-animate'), { once: true });
              }
              var nowUTC = Date.now();
              var diffMs = twilightUTC - nowUTC;
              var sign = '';
              if (diffMs < 0) { diffMs = -diffMs; sign = '-'; }
              var diffTotalMin = Math.floor(diffMs / 60000);
              var hours = Math.floor(diffTotalMin / 60);
              var mins = diffTotalMin % 60;
              var remainingText = '';
              if (sign === '-') {
                remainingText = 'Passed';
              } else if (hours > 0) {
                remainingText = hours + 'h ' + mins + 'min';
              } else {
                remainingText = mins + 'min';
              }
              document.getElementById('twilight-remaining').textContent = 'In ' + remainingText;
            } catch (err) {
              document.getElementById('twilight-time').textContent = '--:-- CLT';
              document.getElementById('twilight-remaining').textContent = '--h --min';
            }
          })
          .catch(err => {
            console.error(err);
            statusDiv.textContent = '⚠️ Forecast file not found or could not be loaded.';
          });
      }

      // initial draw + auto‑reload every 5 minutes
      fetchAndRedraw();
      setInterval(fetchAndRedraw, 5 * 60 * 1000);
  </script>
</body>
</html>
`,
        { headers: { 'Content-Type': 'text/html' } }
      );
    }

    // Serve latest CSV from KV at /api/forecast
    if (url.pathname === '/api/forecast') {
      const latestCSV = await env.TEMP_KV.get('latest_csv');
      if (latestCSV) {
        return new Response(latestCSV, {
          headers: { 'Content-Type': 'text/csv' }
        });
      } else {
        return new Response('No forecast uploaded yet.', { status: 404 });
      }
    }

    if (url.pathname === '/api/forecast_time') {
      const ts = await env.TEMP_KV.get('latest_csv_time');
      return new Response(JSON.stringify({ timestamp: ts }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response('Not found', { status: 404 });
  }
};
