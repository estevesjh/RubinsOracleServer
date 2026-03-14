export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CSV upload (store in KV by source)
    if (request.method === 'POST' && url.pathname === '/api/update') {
      const contentType = request.headers.get('Content-Type');
      if (!contentType || !contentType.includes('text/csv')) {
        return new Response('Invalid content type. Expected text/csv.', { status: 400 });
      }
      const source = url.searchParams.get('source') || 'prophet';
      if (!['prophet', 'nbeats'].includes(source)) {
        return new Response('Invalid source. Use ?source=prophet or ?source=nbeats', { status: 400 });
      }
      const csvData = await request.text();
      await env.TEMP_KV.put('forecast_' + source, csvData);
      const uploadTime =
        request.headers.get('X-Upload-Timestamp') ||
        new Date().toISOString();
      await env.TEMP_KV.put('forecast_' + source + '_time', uploadTime);
      return new Response('CSV uploaded for ' + source + '.', { status: 200 });
    }

    // Serve index.html
    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(
        `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rubin Summit Temperature Forecast</title>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/highcharts.js"><\/script>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/highcharts-more.js"><\/script>
  <script src="https://cdn.jsdelivr.net/npm/highcharts@11/modules/exporting.js"><\/script>
  <style>
.twilight-flex {
  display: flex; flex-direction: row; justify-content: center;
  align-items: flex-start; gap: 2.4rem; margin: 2em auto 0 auto;
  width: 94vw; max-width: 1250px;
}
.twilight-box {
  flex: 1 1 0; max-width: 420px; min-width: 190px;
  background: #b22222; border-radius: 18px;
  box-shadow: 0 6px 32px 0 #0002;
  padding: 1.25em 0.5em 1.1em 0.5em;
  text-align: center; color: #fff;
  margin-top: 0.5em; margin-bottom: 0.5em;
}
.twilight-timebox {
  max-width: 420px; min-width: 190px; background: #555c65;
  border-radius: 18px; box-shadow: 0 2px 16px 0 #0002;
  padding: 1.05em 0.5em 1.0em 0.5em;
  text-align: center; color: #fff;
  margin-top: 1.1em; margin-bottom: 0.5em;
}
.twilight-forecast-uncertainty {
  display: block; font-size: 1.1rem; color: #f5e9e9;
  margin-top: 0.35em; letter-spacing: 0.01em; opacity: 0.93;
}
.twilight-time, .twilight-remaining { color: #fff; }
.twilight-label { color: #ccc; font-size: 1.19rem; margin-bottom: 0.19em; letter-spacing: 0.01em; }
.twilight-temp {
  font-size: 3.3rem; font-family: 'SF Mono','Menlo','Consolas','monospace';
  font-weight: 600; color: #fff; line-height: 1.08; margin-bottom: 0.18em;
}
.twilight-time {
  font-size: 1.8rem; font-family: 'SF Mono','Menlo','Consolas','monospace';
  font-weight: 600; color: #fff; line-height: 1.18; margin-bottom: 0.12em;
}
.twilight-remaining { font-size: 1.31rem; color: #f0f0f0; margin-top: 0.22em; opacity: 0.95; letter-spacing: 0.02em; }
.deg { font-size: 1.2rem; vertical-align: super; margin-left: 2px; color: #ccc; }
.twilight-meta { font-size: 1.07rem; color: #f0f0f0; margin-top: 0.22em; opacity: 0.95; }
#container {
  flex: 2 1 0; min-width: 350px; min-height: 470px; height: 510px; margin: 0;
  background: #18181a; border-radius: 16px; box-shadow: 0 3px 18px 0 #0002;
}
@media (max-width: 1000px) {
  .twilight-flex { flex-direction: column; gap: 1.3rem; width: 99vw; align-items: center; }
  #container { width: 99vw; min-width: 200px; height: 410px; border-radius: 20px;
    box-shadow: 0 3px 18px 0 #0002; padding: 14px; }
}
@keyframes flipY { 0% { transform: rotateX(0deg); } 50% { transform: rotateX(-90deg); } 100% { transform: rotateX(0deg); } }
.flip-animate { display: inline-block; animation: flipY 0.5s ease-in-out; backface-visibility: hidden; }
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
        <div class="twilight-temp" id="twilight-forecast">--<span class="deg">\u00B0C</span></div>
        <span class="twilight-forecast-uncertainty" id="twilight-forecast-uncertainty">\u00B1 -- \u00B0C</span>
        <div class="twilight-meta" id="twilight-actual">Actual: -- \u00B0C (Weather Tower)</div>
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
    var statusDiv = document.createElement('div');
    statusDiv.style.cssText = 'text-align:center;color:red;margin-top:1rem;';
    document.body.prepend(statusDiv);
    var lastUpdateDiv = document.getElementById('lastUpdate');

    function updateBadge(latestTs) {
      function render() {
        var mins = Math.floor(Math.abs(Date.now() - latestTs) / 60000);
        lastUpdateDiv.textContent = 'Last update: ' + mins + ' min ago';
        lastUpdateDiv.style.color = mins > 30 ? 'red' : '#555';
      }
      render();
      if (updateBadge.timer) clearInterval(updateBadge.timer);
      updateBadge.timer = setInterval(render, 60000);
    }

    function updateCurrentTimeCL() {
      var nowCL = new Date().toLocaleTimeString('en-US', {
        timeZone: 'America/Santiago', hour: '2-digit', minute: '2-digit', hour12: false });
      var ct = document.getElementById('current-time');
      if (ct && ct.textContent !== nowCL + ' CLT') {
        ct.classList.add('flip-animate');
        ct.textContent = nowCL + ' CLT';
        ct.addEventListener('animationend', function() { ct.classList.remove('flip-animate'); }, { once: true });
      }
    }
    updateCurrentTimeCL();
    setInterval(updateCurrentTimeCL, 60000);

    var SOURCE_COLORS = {
      prophet: { line: 'firebrick', band: 'rgba(178,34,34,0.25)', label: 'Prophet' },
      nbeats:  { line: 'steelblue', band: 'rgba(70,130,180,0.25)', label: 'NBEATSx' }
    };

    function parseCSV(csv) {
      var NL = String.fromCharCode(10);
      var lines = csv.trim().split(NL);
      var header = lines.shift().split(',');
      var idx = function(n) { return header.indexOf(n); };
      var safe = function(v) { var n = parseFloat(v); return isNaN(n) ? null : +n.toFixed(1); };
      var d = { tempMin: [], tempActual: [], tempMax: [],
                fcMin: [], forecast: [], fcMax: [],
                sunset: [], sunrise: [], source: null, latestPastTs: null };
      lines.forEach(function(l) {
        if (!l.trim()) return;
        var c = l.split(',');
        var ts = new Date(c[idx('timestamp')]).getTime();
        if (ts <= Date.now()) d.latestPastTs = ts;
        if (c[idx('sunset')] && c[idx('sunset')].toLowerCase() === 'true') d.sunset.push(ts);
        if (c[idx('sunrise')] && c[idx('sunrise')].toLowerCase() === 'true') d.sunrise.push(ts);
        d.tempActual.push([ts, safe(c[idx('temp_actual')])]);
        d.forecast.push([ts, safe(c[idx('forecast')])]);
        d.tempMin.push([ts, safe(c[idx('temp_min')])]);
        d.tempMax.push([ts, safe(c[idx('temp_max')])]);
        d.fcMin.push([ts, safe(c[idx('forecast_min')])]);
        d.fcMax.push([ts, safe(c[idx('forecast_max')])]);
        if (!d.source && idx('forecast_source') >= 0) {
          var src = (c[idx('forecast_source')] || '').trim().toLowerCase();
          if (src) d.source = src;
        }
      });
      return d;
    }

    function fetchCSV(source) {
      return fetch('/api/forecast?source=' + source)
        .then(function(r) { return r.ok ? r.text() : null; })
        .then(function(csv) { return csv && csv.trim() ? parseCSV(csv) : null; })
        .catch(function() { return null; });
    }

    function closestIdx(arr, target) {
      return arr.reduce(function(bestIdx, pair, i, a) {
        return Math.abs(pair[0] - target) < Math.abs(a[bestIdx][0] - target) ? i : bestIdx;
      }, 0);
    }

    function fetchAndRedraw() {
      Promise.all([fetchCSV('prophet'), fetchCSV('nbeats')])
        .then(function(results) {
          var primary = results[0] || results[1];
          if (!primary) throw new Error('No forecast data available');

          updateBadge(primary.latestPastTs || primary.forecast[0][0]);

          var obsBand = primary.tempMin.map(function(d, i) { return [d[0], d[1], primary.tempMax[i][1]]; });
          var sunset = primary.sunset, sunrise = primary.sunrise;

          var nightMs = 0;
          if (sunrise.length > 0 && sunset.length > 0) {
            var dayMs = sunset[sunset.length - 1] - sunrise[sunrise.length - 1];
            if (dayMs > 0 && dayMs < 86400000) nightMs = 86400000 - dayMs;
          }
          if (nightMs === 0) nightMs = 12 * 3600 * 1000;

          var nightBands = sunset.map(function(sun) {
            return { color: 'rgba(85, 92, 101, 0.1)', from: sun, to: sun + nightMs,
              label: { text: 'Night Time', style: { color: '#555c65', fontWeight: 600 } }, zIndex: 0 };
          });

          var series = [
            { name: '(max-min)', type: 'arearange', data: obsBand,
              color: '#8080804d', lineWidth: 0, marker: { enabled: false }, zIndex: 0 },
            { name: 'Weather Tower', data: primary.tempActual, color: 'black', zIndex: 2, connectNulls: false }
          ];

          results.forEach(function(d) {
            if (!d) return;
            var src = d.source || 'prophet';
            var pal = SOURCE_COLORS[src] || SOURCE_COLORS.prophet;
            var predBand = d.fcMin.map(function(p, i) { return [p[0], p[1], d.fcMax[i][1]]; });
            var hasBand = predBand.some(function(p) { return p[1] !== null && p[2] !== null; });
            if (hasBand) {
              series.push({ name: pal.label + ' 68% cfi', type: 'arearange', data: predBand,
                color: pal.band, lineWidth: 0, marker: { enabled: false }, zIndex: 0 });
            }
            series.push({ name: pal.label + ' Forecast', data: d.forecast,
              color: pal.line, zIndex: 1, connectNulls: false });
          });

          Highcharts.setOptions({ time: { timezone: 'America/Santiago' } });
          if (window.chart) window.chart.destroy();

          window.chart = Highcharts.chart('container', {
            chart: { type: 'spline', zoomType: 'x', spacing: [40, 40, 20, 20],
                     resetZoomButton: { position: { align: 'right', verticalAlign: 'top', x: 0, y: 0 } } },
            title: { text: null },
            xAxis: { type: 'datetime', title: { text: 'Time (CLT)' }, plotBands: nightBands,
              plotLines: sunset.map(function(ts) {
                return { value: ts, color: 'gray', width: 2, dashStyle: 'Dash',
                  label: { text: 'Sunset', rotation: 90, textAlign: 'left', style: { color: 'gray' } }, zIndex: 5 };
              })
            },
            yAxis: { title: { text: 'Temperature (\\u00B0C)' } },
            tooltip: { shared: true, xDateFormat: '%Y-%m-%d %H:%M',
              formatter: function() {
                var s = '<b>' + Highcharts.dateFormat('%Y-%m-%d %H:%M', this.x) + '</b><br/>';
                this.points.forEach(function(p) {
                  var n = p.series.name, col = p.color;
                  if (n.indexOf('cfi') >= 0 || n === '(max-min)') {
                    if (p.point.high != null && p.point.low != null)
                      s += '<span style="color:' + col + '">\\u25CF</span> ' + n +
                           ': <b>' + (p.point.high - p.point.low).toFixed(2) + '\\u00B0C</b><br/>';
                  } else {
                    if (p.y != null)
                      s += '<span style="color:' + col + '">\\u25CF</span> ' + n +
                           ': <b>' + p.y.toFixed(2) + '\\u00B0C</b><br/>';
                  }
                });
                return s;
              }
            },
            series: series,
            credits: { enabled: false }
          });

          var twilightUTC = sunset.length > 0 ? sunset[sunset.length - 1] : new Date().setHours(21, 0, 0, 0);
          var idxTw = closestIdx(primary.forecast, twilightUTC);
          var forecastTw = primary.forecast[idxTw][1];
          var actualTw = primary.tempActual[idxTw][1];
          var fMinV = primary.fcMin[idxTw][1], fMaxV = primary.fcMax[idxTw][1];
          var uncTw = (fMinV != null && fMaxV != null) ? (fMaxV - fMinV) / 2.0 : NaN;

          document.getElementById('twilight-forecast').innerHTML =
            (forecastTw !== null ? forecastTw.toFixed(1) : '--') + '<span class="deg">\\u00B0C</span>';
          document.getElementById('twilight-forecast-uncertainty').textContent =
            (isFinite(uncTw) && uncTw > 0) ? '\\u00B1 ' + uncTw.toFixed(1) + '\\u00B0C' : '\\u00B1 -- \\u00B0C';
          document.getElementById('twilight-actual').textContent =
            (actualTw !== null ? actualTw.toFixed(1) : '--') + ' \\u00B0C (Weather Tower)';

          try {
            var twCLTime = new Date(twilightUTC).toLocaleTimeString('en-US', {
              timeZone: 'America/Santiago', hour: '2-digit', minute: '2-digit', hour12: false });
            var twElem = document.getElementById('twilight-time');
            if (twElem && twElem.textContent !== twCLTime + ' CLT') {
              twElem.classList.add('flip-animate');
              twElem.textContent = twCLTime + ' CLT';
              twElem.addEventListener('animationend', function() { twElem.classList.remove('flip-animate'); }, { once: true });
            }
            var diffMs = twilightUTC - Date.now();
            var sign = diffMs < 0 ? '-' : '';
            if (diffMs < 0) diffMs = -diffMs;
            var totalMin = Math.floor(diffMs / 60000);
            var hours = Math.floor(totalMin / 60), mins = totalMin % 60;
            document.getElementById('twilight-remaining').textContent =
              sign === '-' ? 'Passed' : 'In ' + (hours > 0 ? hours + 'h ' : '') + mins + 'min';
          } catch (err) {
            document.getElementById('twilight-time').textContent = '--:-- CLT';
            document.getElementById('twilight-remaining').textContent = '--h --min';
          }
        })
        .catch(function(err) {
          console.error(err);
          statusDiv.textContent = '\\u26A0\\uFE0F Forecast file not found or could not be loaded.';
        });
    }

    fetchAndRedraw();
    setInterval(fetchAndRedraw, 5 * 60 * 1000);
  <\/script>
</body>
</html>
`,
        { headers: { 'Content-Type': 'text/html' } }
      );
    }

    // Serve CSV from KV: /api/forecast?source=prophet|nbeats (default: prophet)
    if (url.pathname === '/api/forecast') {
      const source = url.searchParams.get('source') || 'prophet';
      const csv = await env.TEMP_KV.get('forecast_' + source);
      if (csv) {
        return new Response(csv, { headers: { 'Content-Type': 'text/csv' } });
      } else {
        return new Response('No forecast for ' + source + '.', { status: 404 });
      }
    }

    if (url.pathname === '/api/forecast_time') {
      const source = url.searchParams.get('source') || 'prophet';
      const ts = await env.TEMP_KV.get('forecast_' + source + '_time');
      return new Response(JSON.stringify({ timestamp: ts }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response('Not found', { status: 404 });
  }
};
