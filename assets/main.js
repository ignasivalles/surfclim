// main.js — surfclim dashboard: stat cards + Leaflet map
(function () {
  'use strict';

  var CSV_PATH        = 'data/individual_data.csv';
  var MHW_JSON_PATH   = 'data/mhw_status.json';
  var MAP_CENTER      = [43.55, -4.0];
  var MAP_ZOOM        = 10;
  var MAP_ZOOM_MOBILE = 9;
  var MOBILE_BP       = 768;
  var COLOR_MIN       = -3;
  var COLOR_MAX       = 3;

  /* ── Highlight Map nav pill when section is in viewport ── */
  var mapSection = document.getElementById('map-section');
  var mapPill    = document.querySelector('.nav-pill[href="#map-section"]');

  if (mapSection && mapPill) {
    var observer = new IntersectionObserver(function (entries) {
      mapPill.classList.toggle('active', entries[0].isIntersecting);
    }, { threshold: 0.1 });
    observer.observe(mapSection);
  }

  /* ── Anomaly → color (blue → white → red) ── */
  function getColor(anomaly) {
    var v = parseFloat(anomaly);
    if (!isFinite(v)) v = 0;
    var mid = (COLOR_MIN + COLOR_MAX) / 2;
    var ratio, r, g, b;
    if (v < mid) {
      ratio = Math.max(0, Math.min(1, (v - COLOR_MIN) / (mid - COLOR_MIN)));
      r = Math.round(ratio * 255);
      g = Math.round(ratio * 255);
      b = 255;
    } else {
      ratio = Math.max(0, Math.min(1, (v - mid) / (COLOR_MAX - mid)));
      r = 255;
      g = Math.round(255 - ratio * 255);
      b = Math.round(255 - ratio * 255);
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  /* ── MHW thresholds — loaded from cmems_climatology.json ─────────
     Falls back to hardcoded 1970-78 values if the JSON is unavailable.  */
  var MHW_P90   = null;  // populated after fetch
  var MHW_DELTA = null;
  var MHW_SOURCE = '1991–2020 (CMEMS)';

  // Fallback: 1970-78 Sardinero/Magdalena baseline
  var MHW_P90_FALLBACK   = {1:12.31,2:12.14,3:13.00,4:13.50,5:15.50,6:17.10,
                             7:19.90,8:20.80,9:20.00,10:18.00,11:15.80,12:13.99};
  var MHW_DELTA_FALLBACK = {1:1.41,2:1.14,3:1.60,4:1.05,5:1.70,6:1.10,
                             7:1.95,8:1.40,9:1.75,10:2.00,11:2.10,12:1.99};

  function mhwCategory(temp, month) {
    var p90   = (MHW_P90   || MHW_P90_FALLBACK)[month];
    var delta = (MHW_DELTA || MHW_DELTA_FALLBACK)[month];
    if (temp >= p90 + 3 * delta) return 'Extreme';
    if (temp >= p90 + 2 * delta) return 'Severe';
    if (temp >= p90 +     delta) return 'Strong';
    if (temp >= p90)              return 'Moderate';
    return 'None';
  }

  var MHW_STYLE = {
    'None':     { color: '#27ae60', bg: '' },
    'Moderate': { color: '#f39c12', bg: 'rgba(243,156,18,0.06)' },
    'Strong':   { color: '#e67e22', bg: 'rgba(230,126,34,0.07)' },
    'Severe':   { color: '#e74c3c', bg: 'rgba(231,76,60,0.07)' },
    'Extreme':  { color: '#8e1c1c', bg: 'rgba(142,28,28,0.09)' },
  };

  function fillMhwCard(category, temp, p90, dateLabel) {
    var card    = document.getElementById('card-mhw');
    var valueEl = document.getElementById('card-mhw-value');
    var subEl   = document.getElementById('card-mhw-sub');
    var dateEl  = document.getElementById('card-mhw-date');
    if (!card) return;

    var s = MHW_STYLE[category] || MHW_STYLE['None'];
    valueEl.textContent    = category;
    valueEl.style.color    = s.color;
    valueEl.style.fontSize = '1.35rem';

    var diff = (temp - p90);
    subEl.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(1) + '°C vs p90 (' + p90.toFixed(1) + '°C)';
    subEl.style.color = s.color;

    dateEl.textContent = dateLabel;
    if (s.bg) card.style.background = s.bg;
    card.style.borderLeft = '3px solid ' + s.color;
  }

  // Load CMEMS thresholds, then trigger MHW card if CSV already processed
  var _mhwPending = null;  // holds {temp, month, p90Fallback, dateLabel} until thresholds ready
  fetch(MHW_JSON_PATH)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      MHW_P90   = {};
      MHW_DELTA = {};
      for (var m = 1; m <= 12; m++) {
        MHW_P90[m]   = data[String(m)].p90;
        MHW_DELTA[m] = data[String(m)].delta;
      }
      MHW_SOURCE = (data._meta || {}).period || '1991–2020';
      if (_mhwPending) {
        var p = _mhwPending;
        fillMhwCard(mhwCategory(p.temp, p.month), p.temp, MHW_P90[p.month], p.dateLabel);
        _mhwPending = null;
      }
    })
    .catch(function () { /* use fallback values already in MHW_P90_FALLBACK */ });

  /* ── Stat cards ─────────────────────────────────────────────────── */
  var MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec'];

  function mean(arr) {
    return arr.reduce(function (s, v) { return s + v; }, 0) / arr.length;
  }
  function fmtAnom(v) { return (v >= 0 ? '+' : '') + v.toFixed(1) + '\u00b0C vs 70s'; }
  function anomColor(v) {
    if (v > 0.3)  return '#e74c3c';
    if (v < -0.3) return '#3498db';
    return '';
  }
  function fillCard(valueId, subId, dateId, temp, anom, dateLabel) {
    document.getElementById(valueId).textContent = temp.toFixed(1) + '\u00b0C';
    var subEl = document.getElementById(subId);
    subEl.textContent = fmtAnom(anom);
    var c = anomColor(anom);
    if (c) subEl.style.color = c;
    document.getElementById(dateId).textContent = dateLabel;
  }

  /* ── Map ─────────────────────────────────────────────────────────── */
  var map         = null;
  var markerLayer = null;
  var allMarkers  = [];

  function initMap() {
    var isMobile = window.innerWidth < MOBILE_BP;
    map = L.map('map').setView(MAP_CENTER, isMobile ? MAP_ZOOM_MOBILE : MAP_ZOOM);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19
    }).addTo(map);

    markerLayer = L.layerGroup().addTo(map);

    // Color bar legend
    var colorBar = L.control({ position: 'bottomleft' });
    colorBar.onAdd = function () {
      var div = L.DomUtil.create('div', 'color-bar-container');
      div.innerHTML = '<div class="color-bar-title">Anomaly vs 1970s ref.</div>' +
                      '<div class="color-bar"></div>' +
                      '<div class="color-bar-labels">' +
                      '<span>-3\u00b0C</span><span>0\u00b0C</span><span>+3\u00b0C</span>' +
                      '</div>';
      return div;
    };
    colorBar.addTo(map);

    // Month filter
    var dropdown = L.control({ position: 'bottomleft' });
    dropdown.onAdd = function () {
      var div = L.DomUtil.create('div', 'dropdown-container');
      div.innerHTML = '<select id="month-select" aria-label="Filter by month">' +
        '<option value="all">All months</option>' +
        '<option value="01">January</option><option value="02">February</option>' +
        '<option value="03">March</option><option value="04">April</option>' +
        '<option value="05">May</option><option value="06">June</option>' +
        '<option value="07">July</option><option value="08">August</option>' +
        '<option value="09">September</option><option value="10">October</option>' +
        '<option value="11">November</option><option value="12">December</option>' +
        '</select>';
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    dropdown.addTo(map);

    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === 'month-select') filterMarkers();
    });
  }

  function filterMarkers() {
    var sel   = document.getElementById('month-select');
    var month = sel ? sel.value : 'all';
    markerLayer.clearLayers();
    allMarkers.forEach(function (obj) {
      if (month === 'all' || obj.month === month) markerLayer.addLayer(obj.marker);
    });
  }

  function getMonthStr(dateStr) {
    if (!dateStr) return '';
    var m = dateStr.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return m[2];
    var d = new Date(dateStr);
    return isNaN(d) ? '' : String(d.getMonth() + 1).padStart(2, '0');
  }

  /* ── Load CSV → stat cards + map markers ─────────────────────────── */
  fetch(CSV_PATH)
    .then(function (r) { return r.text(); })
    .then(function (csvText) {
      Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
          if (!results || !results.data) return;

          var rows = results.data.filter(function (row) {
            var lat = parseFloat(row.Latitude);
            var lon = parseFloat(row.Longitude);
            return lon >= -5 && lon <= -3 && lat >= 43.2 && lat <= 43.9 &&
                   isFinite(parseFloat(row.Temperature)) &&
                   isFinite(parseFloat(row.Temperature_Anomaly));
          }).sort(function (a, b) {
            return new Date(a.Date) - new Date(b.Date);
          });

          if (!rows.length) return;

          /* ─ Stat cards ─ */
          var latest     = rows[rows.length - 1];
          var latestDate = new Date(latest.Date);
          var lM = latestDate.getMonth();
          var lY = latestDate.getFullYear();

          var monthRows = rows.filter(function (r) {
            var d = new Date(r.Date);
            return d.getMonth() === lM && d.getFullYear() === lY;
          });
          var yearRows = rows.filter(function (r) {
            return new Date(r.Date).getFullYear() === lY;
          });

          fillCard('card1-value', 'card1-sub', 'card1-date',
                   parseFloat(latest.Temperature),
                   parseFloat(latest.Temperature_Anomaly),
                   latestDate.toISOString().slice(0, 10));

          if (monthRows.length) {
            fillCard('card2-value', 'card2-sub', 'card2-date',
                     mean(monthRows.map(function (r) { return parseFloat(r.Temperature); })),
                     mean(monthRows.map(function (r) { return parseFloat(r.Temperature_Anomaly); })),
                     MONTHS[lM] + ' ' + lY);
          }

          if (yearRows.length) {
            fillCard('card3-value', 'card3-sub', 'card3-date',
                     mean(yearRows.map(function (r) { return parseFloat(r.Temperature); })),
                     mean(yearRows.map(function (r) { return parseFloat(r.Temperature_Anomaly); })),
                     String(lY));
          }

          /* ─ MHW card ─ */
          var latestTemp  = parseFloat(latest.Temperature);
          var latestMonth = latestDate.getMonth() + 1;
          var dateLabel   = latestDate.toISOString().slice(0, 10);
          if (MHW_P90) {
            // CMEMS thresholds already loaded
            fillMhwCard(mhwCategory(latestTemp, latestMonth),
                        latestTemp, MHW_P90[latestMonth], dateLabel);
          } else {
            // Thresholds still loading — stash and fill once ready
            _mhwPending = { temp: latestTemp, month: latestMonth, dateLabel: dateLabel };
            // Immediate render with fallback
            fillMhwCard(mhwCategory(latestTemp, latestMonth),
                        latestTemp, MHW_P90_FALLBACK[latestMonth], dateLabel);
          }

          /* ─ Map markers ─ */
          rows.forEach(function (row) {
            var lat   = parseFloat(row.Latitude);
            var lon   = parseFloat(row.Longitude);
            var temp  = parseFloat(row.Temperature).toFixed(1);
            var anom  = parseFloat(row.Temperature_Anomaly);
            var date  = (row.Date || '').slice(0, 10);
            var month = getMonthStr(row.Date);
            var color = getColor(anom);

            var marker = L.circleMarker([lat, lon], {
              radius: 6,
              color: 'rgba(0,0,0,0.25)',
              weight: 1,
              fillColor: color,
              fillOpacity: 0.75
            });

            marker.bindPopup(
              '<div class="data-popup">' +
              '<strong>Temperature:</strong> ' + temp + '\u00b0C<br>' +
              '<strong>Anomaly:</strong> ' + (anom >= 0 ? '+' : '') + anom.toFixed(1) + '\u00b0C<br>' +
              '<strong>Date:</strong> ' + date +
              '</div>',
              { maxWidth: 220 }
            );

            allMarkers.push({ marker: marker, month: month });
            markerLayer.addLayer(marker);
          });
        }
      });
    })
    .catch(function () { /* stat cards stay at '—' */ });

  /* ── Init ── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMap);
  } else {
    initMap();
  }

})();
