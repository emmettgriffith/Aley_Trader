(function () {
  const data = window.__INITIAL_DATA__ || { symbol: 'SYMB', bars: [] };
  const theme = Object.assign({
    palette: {
      primary_bg: '#050E1B',
      secondary_bg: '#0F1F33',
      text_primary: '#F5FAFF',
      text_secondary: '#C7D3E3',
      positive: '#24E1A6',
      negative: '#FF5C8A',
      info: '#5DADE2'
    }
  }, data.theme || {});

  const state = {
    overlays: {
      ma: true,
      bb: false,
      rsi: true,
      vwma: false
    },
    activeTool: 'cursor',
    priceLines: []
  };

  const dom = {
    summary: document.getElementById('symbol-summary'),
    watchlist: document.getElementById('watchlist'),
    status: document.getElementById('status-strip'),
    pills: Array.from(document.querySelectorAll('.pill')),
    toolbarButtons: Array.from(document.querySelectorAll('.toolbar-btn')),
    timeframes: Array.from(document.querySelectorAll('.timeframes button')),
    exportBtn: document.getElementById('export-chart'),
    toggleLights: document.getElementById('dark-light-toggle')
  };

  const formatNumber = (value, digits = 2) => {
    if (value === null || value === undefined || isNaN(value)) {
      return '—';
    }
    if (Math.abs(value) >= 1e9) {
      return (value / 1e9).toFixed(digits) + 'B';
    }
    if (Math.abs(value) >= 1e6) {
      return (value / 1e6).toFixed(digits) + 'M';
    }
    if (Math.abs(value) >= 1e3) {
      return (value / 1e3).toFixed(digits) + 'K';
    }
    return Number(value).toFixed(digits);
  };

  const formatPercent = (value) => {
    if (value === null || value === undefined || isNaN(value)) {
      return '—';
    }
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const buildSummary = () => {
    if (!dom.summary) return;
    const meta = data.meta || {};
    const latest = meta.lastPrice || (data.bars[data.bars.length - 1] || {}).close;
    const change = meta.changePercent ?? meta.change ?? 0;
    const isNegative = change < 0;
    dom.summary.innerHTML = `
      <div class="symbol">
        <span>${data.symbol || 'SYMB'}</span>
        <small>${meta.exchange || ''}</small>
      </div>
      <div class="price">${formatNumber(latest, latest >= 100 ? 2 : 4)}</div>
      <div class="change ${isNegative ? 'negative' : 'positive'}">
        ${formatPercent(change)}
      </div>
      <div class="meta">
        <span>${data.company || ''}</span>
        <span>${meta.session || 'REGULAR'}</span>
      </div>
    `;
  };

  const buildWatchlist = () => {
    if (!dom.watchlist) return;
    const entries = data.watchlist || [];
    if (!entries.length) {
      dom.watchlist.innerHTML = '<p class="empty">No watchlist entries</p>';
      return;
    }
    const rows = entries.map((item) => {
      const negative = item.change < 0;
      return `
        <div class="watch-row">
          <div class="ticker">
            <strong>${item.symbol}</strong>
            <small>${item.name || ''}</small>
          </div>
          <div class="last">${formatNumber(item.price, item.price >= 100 ? 2 : 4)}</div>
          <div class="change ${negative ? 'negative' : 'positive'}">
            ${formatPercent(item.change_percent ?? item.change)}
          </div>
        </div>
      `;
    });
    dom.watchlist.innerHTML = rows.join('');
  };

  const buildStatusStrip = () => {
    if (!dom.status) return;
    const stats = data.stats || {};
    dom.status.innerHTML = `
      <span>ATR <strong>${formatNumber(stats.atr || 0, 2)}</strong></span>
      <span>High <strong>${formatNumber(stats.session_high || 0, 2)}</strong></span>
      <span>Low <strong>${formatNumber(stats.session_low || 0, 2)}</strong></span>
      <span>Volume <strong>${formatNumber(stats.volume || 0, 2)}</strong></span>
      <span>Range <strong>${formatNumber(stats.range || 0, 2)}</strong></span>
    `;
  };

  buildSummary();
  buildWatchlist();
  buildStatusStrip();

  const createChart = LightweightCharts.createChart;
  const primaryChart = createChart(document.getElementById('primary-chart'), {
    layout: {
      background: { color: 'transparent' },
      textColor: theme.palette.text_secondary
    },
    grid: {
      vertLines: { color: 'rgba(76, 199, 255, 0.08)' },
      horzLines: { color: 'rgba(76, 199, 255, 0.08)' }
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: 'rgba(120, 199, 255, 0.35)' },
      horzLine: { color: 'rgba(120, 199, 255, 0.35)' }
    },
    timeScale: {
      borderVisible: false,
      rightOffset: 12,
      barSpacing: 9,
      ticksVisible: true
    },
    rightPriceScale: {
      borderVisible: false
    },
    handleScroll: true,
    handleScale: true,
    autoSize: true
  });

  const candleSeries = primaryChart.addCandlestickSeries({
    upColor: theme.palette.positive,
    downColor: theme.palette.negative,
    borderUpColor: theme.palette.positive,
    borderDownColor: theme.palette.negative,
    wickUpColor: theme.palette.positive,
    wickDownColor: theme.palette.negative
  });

  candleSeries.setData(data.bars || []);

  const overlaySeries = {
    maFast: primaryChart.addLineSeries({
      color: '#76C7FF',
      lineWidth: 2,
      priceScaleId: 'right',
      title: 'MA 21'
    }),
    maSlow: primaryChart.addLineSeries({
      color: '#F7B731',
      lineWidth: 2,
      priceScaleId: 'right',
      title: 'MA 50'
    }),
    vwma: primaryChart.addLineSeries({
      color: '#29E3FF',
      lineWidth: 2,
      priceScaleId: 'right',
      title: 'VWMA'
    }),
    bbUpper: primaryChart.addLineSeries({
      color: 'rgba(118,199,255,0.55)',
      lineWidth: 1,
      priceScaleId: 'right'
    }),
    bbLower: primaryChart.addLineSeries({
      color: 'rgba(118,199,255,0.55)',
      lineWidth: 1,
      priceScaleId: 'right'
    })
  };

  const volumeChart = createChart(document.getElementById('volume-chart'), {
    layout: {
      background: { color: 'transparent' },
      textColor: theme.palette.text_secondary
    },
    grid: {
      vertLines: { color: 'rgba(76, 199, 255, 0.04)' },
      horzLines: { color: 'rgba(76, 199, 255, 0.04)' }
    },
    timeScale: {
      visible: true,
      borderVisible: false
    },
    rightPriceScale: {
      borderVisible: false
    },
    autoSize: true,
    height: 160
  });

  const volumeSeries = volumeChart.addHistogramSeries({
    color: 'rgba(118, 199, 255, 0.6)',
    priceFormat: { type: 'volume' },
    priceScaleId: ''
  });
  volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.2, bottom: 0 } });
  volumeSeries.setData(data.volume || []);

  const oscillatorChart = createChart(document.getElementById('oscillator-chart'), {
    layout: {
      background: { color: 'transparent' },
      textColor: theme.palette.text_secondary
    },
    grid: {
      vertLines: { color: 'rgba(76, 199, 255, 0.04)' },
      horzLines: { color: 'rgba(76, 199, 255, 0.04)' }
    },
    rightPriceScale: {
      borderVisible: false
    },
    timeScale: {
      visible: true,
      borderVisible: false
    },
    handleScale: true,
    autoSize: true,
    height: 160
  });

  const rsiSeries = oscillatorChart.addLineSeries({
    color: '#5DADE2',
    lineWidth: 2,
    priceScaleId: 'right',
    title: 'RSI 14'
  });
  rsiSeries.setData(((data.oscillators || {}).rsi) || []);

  const overboughtLine = oscillatorChart.addLineSeries({
    color: 'rgba(255, 108, 147, 0.4)',
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    priceScaleId: 'right'
  });
  const oversoldLine = oscillatorChart.addLineSeries({
    color: 'rgba(76, 211, 182, 0.4)',
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    priceScaleId: 'right'
  });
  const rsiLevels = (data.oscillators && data.oscillators.rsi_levels) || {};
  const rsiTimes = (data.oscillators && data.oscillators.rsi) ? data.oscillators.rsi.map(point => ({ time: point.time })) : [];
  overboughtLine.setData(rsiTimes.map(item => ({ time: item.time, value: rsiLevels.overbought || 70 })));
  oversoldLine.setData(rsiTimes.map(item => ({ time: item.time, value: rsiLevels.oversold || 30 })));

  const syncTimescales = () => {
    const primaryTime = primaryChart.timeScale();
    const volumeTime = volumeChart.timeScale();
    const oscTime = oscillatorChart.timeScale();

    primaryTime.subscribeVisibleTimeRangeChange(range => {
      if (!range) return;
      volumeTime.setVisibleRange(range);
      oscTime.setVisibleRange(range);
    });
  };
  syncTimescales();

  const applyOverlayVisibility = () => {
    if (!data.overlays) return;
    if (data.overlays.ma_fast) {
      overlaySeries.maFast.setData(state.overlays.ma ? data.overlays.ma_fast : []);
    }
    if (data.overlays.ma_slow) {
      overlaySeries.maSlow.setData(state.overlays.ma ? data.overlays.ma_slow : []);
    }
    if (data.overlays.vwma) {
      overlaySeries.vwma.setData(state.overlays.vwma ? data.overlays.vwma : []);
    }
    if (data.overlays.bb_upper && data.overlays.bb_lower) {
      overlaySeries.bbUpper.setData(state.overlays.bb ? data.overlays.bb_upper : []);
      overlaySeries.bbLower.setData(state.overlays.bb ? data.overlays.bb_lower : []);
    }
    rsiSeries.applyOptions({ visible: state.overlays.rsi });
    oscillatorChart.applyOptions({
      priceScale: { visible: state.overlays.rsi }
    });
    document.getElementById('oscillator-chart').style.display = state.overlays.rsi ? 'block' : 'none';
  };
  applyOverlayVisibility();

  dom.pills.forEach((pill) => {
    pill.addEventListener('click', () => {
      const key = pill.dataset.overlay;
      const next = !state.overlays[key];
      state.overlays[key] = next;
      pill.classList.toggle('active', next);
      applyOverlayVisibility();
    });
  });

  dom.toolbarButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'reset') {
        state.priceLines.forEach(line => {
          try { line.remove(); } catch (err) {}
        });
        state.priceLines = [];
        primaryChart.timeScale().fitContent();
        return;
      }
      if (action === 'note') {
        toggleAnnotationPanel();
        return;
      }
      dom.toolbarButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeTool = action;
    });
  });

  const chartContainer = document.getElementById('primary-chart');
  let annotationPanel;
  const toggleAnnotationPanel = () => {
    if (!annotationPanel) {
      annotationPanel = document.createElement('div');
      annotationPanel.className = 'annotation-panel';
      annotationPanel.innerHTML = `
        <h4>Chart Note</h4>
        <textarea placeholder="Type a note..." aria-label="Annotation text"></textarea>
        <div style="display:flex;gap:8px;margin-top:10px;">
          <button id="save-note">Save</button>
          <button id="close-note">Close</button>
        </div>
        <p style="margin-top:8px;color:var(--text-muted);font-size:0.8rem;">Notes persist in this session.</p>
      `;
      chartContainer.appendChild(annotationPanel);
      annotationPanel.querySelector('#close-note').addEventListener('click', () => {
        annotationPanel.classList.remove('open');
      });
      annotationPanel.querySelector('#save-note').addEventListener('click', () => {
        annotationPanel.classList.remove('open');
      });
    }
    annotationPanel.classList.toggle('open');
  };

  primaryChart.subscribeClick((params) => {
    if (!params || params.point === undefined) return;
    const price = params.price;
    if (!price || state.activeTool === 'cursor') {
      return;
    }
    if (state.activeTool === 'trendline' || state.activeTool === 'horizontal') {
      const priceLine = candleSeries.createPriceLine({
        price,
        color: state.activeTool === 'horizontal' ? 'rgba(118,199,255,0.9)' : 'rgba(41,227,255,0.9)',
        lineWidth: 1,
        lineStyle: state.activeTool === 'trendline' ? LightweightCharts.LineStyle.Dotted : LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: `${state.activeTool === 'trendline' ? 'Alert' : 'Horizontal'} @ ${formatNumber(price, 2)}`
      });
      state.priceLines.push(priceLine);
      if (state.activeTool === 'trendline') {
        setTimeout(() => {
          dom.toolbarButtons.forEach(b => b.dataset.action === 'cursor' && b.click());
        }, 0);
      }
    }
  });

  const crosshairFormatter = document.createElement('div');
  crosshairFormatter.style.position = 'absolute';
  crosshairFormatter.style.pointerEvents = 'none';
  crosshairFormatter.style.padding = '10px 12px';
  crosshairFormatter.style.borderRadius = '12px';
  crosshairFormatter.style.background = 'rgba(5, 14, 27, 0.92)';
  crosshairFormatter.style.border = '1px solid rgba(118, 199, 255, 0.2)';
  crosshairFormatter.style.boxShadow = 'var(--shadow)';
  crosshairFormatter.style.fontSize = '0.82rem';
  crosshairFormatter.style.display = 'none';
  chartContainer.appendChild(crosshairFormatter);

  primaryChart.subscribeCrosshairMove((param) => {
    if (!param || !param.time || !param.point) {
      crosshairFormatter.style.display = 'none';
      return;
    }
    const price = param.seriesPrices.get(candleSeries);
    if (!price) {
      crosshairFormatter.style.display = 'none';
      return;
    }
    crosshairFormatter.style.display = 'block';
    crosshairFormatter.style.left = `${param.point.x + 12}px`;
    crosshairFormatter.style.top = `${param.point.y + 12}px`;
    crosshairFormatter.innerHTML = `
      <strong>${data.symbol}</strong>
      <div>O ${formatNumber(price.open)}</div>
      <div>H ${formatNumber(price.high)}</div>
      <div>L ${formatNumber(price.low)}</div>
      <div>C ${formatNumber(price.close)}</div>
    `;
  });

  dom.timeframes.forEach((btn) => {
    btn.addEventListener('click', () => {
      dom.timeframes.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tf = btn.dataset.timeframe;
      dom.status.insertAdjacentHTML('beforeend', `<span>Switched to <strong>${tf}</strong></span>`);
      setTimeout(() => {
        const spans = dom.status.querySelectorAll('span');
        if (spans.length > 5) {
          spans[spans.length - 1].remove();
        }
      }, 4000);
    });
  });

  if (dom.exportBtn) {
    dom.exportBtn.addEventListener('click', async () => {
      try {
        const bitmap = await primaryChart.takeScreenshot();
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `${data.symbol || 'chart'}_${Date.now()}.png`;
        link.click();
      } catch (error) {
        console.error('Export failed', error);
      }
    });
  }

  if (dom.toggleLights) {
    dom.toggleLights.addEventListener('click', () => {
      document.body.classList.toggle('light-mode');
      dom.toggleLights.classList.toggle('active');
    });
  }

  // Align the volume and oscillator charts with the main chart width
  const resizeObserver = new ResizeObserver(() => {
    const { width, height } = document.getElementById('primary-chart').getBoundingClientRect();
    primaryChart.resize(width, height);
    const volumeBox = document.getElementById('volume-chart').getBoundingClientRect();
    volumeChart.resize(volumeBox.width, volumeBox.height);
    const oscBox = document.getElementById('oscillator-chart').getBoundingClientRect();
    oscillatorChart.resize(oscBox.width, oscBox.height);
  });
  resizeObserver.observe(document.getElementById('primary-chart'));

})();
