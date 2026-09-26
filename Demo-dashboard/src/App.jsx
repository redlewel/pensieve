import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDownWideNarrow, ArrowRight, Braces, ChevronDown, CircleHelp, Clock3, Command, Database, FileCode2, Filter, History, Landmark, Layers3, Search, Sparkles, SlidersHorizontal, Wallet } from 'lucide-react';
import { API_BASE_URL, getProjectSchema, rawRecall, recall } from './api.js';

const projects = [
  { id: 'finance', label: 'Personal Wealth', dimension: 'financial_impact', categories: ['spending', 'investment', 'tax', 'income', 'debt'] },
  { id: 'engineering', label: 'Project Historian', dimension: 'architectural_impact', categories: ['outage', 'refactor', 'decision', 'roadmap'] },
  { id: 'autobiography', label: 'Life Autobiographer', dimension: 'emotional_significance', categories: ['career', 'family', 'travel', 'health'] },
];
const mockDatabase = {
  finance: [
    { _id: 'mock-fin-1', text: 'Bought a $4 iced coffee at Starbucks.', timestamp: '2024-11-10T09:15:00Z', categories: ['spending'], domain_weights: { financial_impact: 0.1, liquidity_risk: 0.1, tax_relevance: 0 }, sim: 0.92, access_count: 0 },
    { _id: 'mock-fin-2', text: 'Found a $5 bill in my winter coat pocket.', timestamp: '2024-01-15T10:00:00Z', categories: ['spending'], domain_weights: { financial_impact: 0.1, liquidity_risk: 0, tax_relevance: 0 }, sim: 0.88, access_count: 0 },
    { _id: 'mock-fin-3', text: 'Transferred $50,000 into my Vanguard Index Fund.', timestamp: '2023-08-20T14:00:00Z', categories: ['investment'], domain_weights: { financial_impact: 9.8, liquidity_risk: 5, tax_relevance: 4 }, sim: 0.75, access_count: 0 },
    { _id: 'mock-fin-4', text: 'Signed a lease for a new apartment at $3,200/month.', timestamp: '2024-05-01T12:00:00Z', categories: ['debt', 'spending'], domain_weights: { financial_impact: 8.5, liquidity_risk: 7, tax_relevance: 0 }, sim: 0.65, access_count: 0 },
    { _id: 'mock-fin-5', text: 'Paid $12 for a parking ticket downtown.', timestamp: '2024-10-25T16:00:00Z', categories: ['spending'], domain_weights: { financial_impact: 0.5, liquidity_risk: 0.8, tax_relevance: 0 }, sim: 0.85, access_count: 0 },
    { _id: 'mock-fin-6', text: 'Negotiated a 12% salary increase after my annual review.', timestamp: '2024-06-12T11:00:00Z', categories: ['income'], domain_weights: { financial_impact: 8.4, liquidity_risk: 2, tax_relevance: 2 }, sim: 0.68, access_count: 0 },
  ],
  engineering: [
    { _id: 'mock-eng-1', text: 'Migrated the production database to MongoDB.', timestamp: '2025-04-18T15:00:00Z', categories: ['decision', 'roadmap'], domain_weights: { architectural_impact: 9.5, strategic_risk: 7, team_dependency: 6 }, sim: 0.81, access_count: 0 },
    { _id: 'mock-eng-2', text: 'Updated the settings button padding from 12px to 14px.', timestamp: '2025-06-01T10:00:00Z', categories: ['refactor'], domain_weights: { architectural_impact: 0.5, strategic_risk: 0, team_dependency: 0.2 }, sim: 0.91, access_count: 0 },
    { _id: 'mock-eng-3', text: 'The payment service outage delayed checkout for 40 minutes.', timestamp: '2025-02-20T08:30:00Z', categories: ['outage'], domain_weights: { architectural_impact: 7, strategic_risk: 9.2, team_dependency: 8 }, sim: 0.74, access_count: 0 },
  ],
  autobiography: [
    { _id: 'mock-life-1', text: 'Got married in Tuscany surrounded by family.', timestamp: '2024-09-14T14:00:00Z', categories: ['family', 'travel'], domain_weights: { emotional_significance: 9.9, relationship_impact: 9, life_milestone_tier: 10 }, sim: 0.83, access_count: 0 },
    { _id: 'mock-life-2', text: 'Started a new role after a long career search.', timestamp: '2025-03-03T09:00:00Z', categories: ['career'], domain_weights: { emotional_significance: 7.5, relationship_impact: 2, life_milestone_tier: 6 }, sim: 0.74, access_count: 0 },
    { _id: 'mock-life-3', text: 'Watched television after work and went to bed early.', timestamp: '2025-05-19T21:00:00Z', categories: ['health'], domain_weights: { emotional_significance: 0.3, relationship_impact: 0.2, life_milestone_tier: 0.1 }, sim: 0.9, access_count: 0 },
  ],
};
const initialWeights = { similarity: 60 };
const resultLimit = 20;
const formatDate = (date) => date ? new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Date unknown';
const defaultQuery = "How's my spending?";
const scoreValue = (item, weighted, scoreOverride) => weighted ? item.final_score : scoreOverride ?? item.similarity ?? item.sim ?? item.score_breakdown?.similarity ?? 0;
const readable = (key) => key.replaceAll('_', ' ');
const toIso = (localDateTime) => localDateTime ? new Date(localDateTime).toISOString() : undefined;

function mockResults(project, dimensions, schema, dimensionBiases, dateRange) {
  const from = dateRange.since ? new Date(dateRange.since).getTime() : -Infinity;
  const to = dateRange.until ? new Date(dateRange.until).getTime() : Infinity;
  const candidates = (mockDatabase[project] || []).filter(item => {
    const time = new Date(item.timestamp).getTime();
    return time >= from && time <= to;
  });
  const naive = [...candidates].sort((a, b) => b.sim - a.sim).slice(0, 5);
  const weighted = candidates.map(item => {
    const ageDays = Math.max(0, (Date.now() - new Date(item.timestamp).getTime()) / 86400000);
    const dimensionBreakdowns = Object.fromEntries(dimensions.map(dimension => {
      const weight = Number(item.domain_weights[dimension] ?? schema?.dimensions?.[dimension]?.default ?? 0);
      const halfLife = Number(schema?.dimensions?.[dimension]?.decay_half_life_days ?? 365);
      const decay = Math.exp(-ageDays / halfLife);
      const bias = (dimensionBiases[dimension] ?? 30) / 100 * 0.5;
      return [dimension, { weight, decay, contribution: weight * bias * decay }];
    }));
    const similarity = item.sim;
    const domainScore = Object.values(dimensionBreakdowns).reduce((sum, part) => sum + part.contribution, 0);
    return { ...item, age_days: ageDays, dimension_breakdowns: dimensionBreakdowns, final_score: similarity + domainScore, score_breakdown: { similarity: item.sim } };
  }).sort((a, b) => b.final_score - a.final_score).slice(0, resultLimit);
  return { naive, weighted };
}

function mockNaiveResults(project, dateRange) {
  const from = dateRange.since ? new Date(dateRange.since).getTime() : -Infinity;
  const to = dateRange.until ? new Date(dateRange.until).getTime() : Infinity;
  return (mockDatabase[project] || []).filter(item => {
    const time = new Date(item.timestamp).getTime();
    return time >= from && time <= to;
  }).sort((a, b) => b.sim - a.sim).slice(0, 5);
}

function LoadingSkeletons({ count = 4, weighted = false }) {
  return Array.from({ length: count }, (_, index) => <article className={`event-card skeleton-card ${weighted ? 'weighted-card' : 'naive-card'}`} key={index} aria-hidden="true">
    <div className="skeleton-line skeleton-meta" />
    <div className="skeleton-line skeleton-copy" />
    <div className="skeleton-line skeleton-copy short" />
    <div className="skeleton-bottom"><span className="skeleton-line skeleton-score" /><span className="skeleton-line skeleton-track" /></div>
    {weighted && <div className="skeleton-line skeleton-breakdown" />}
  </article>);
}

function DimensionSliderSkeletons() {
  return <div className="dimension-sliders schema-skeletons" aria-label="Loading project dimensions">
    {[0, 1, 2].map(index => <div className="slider-row slider-skeleton" key={index} aria-hidden="true"><div className="slider-top"><span className="skeleton-line skeleton-icon" /><span className="skeleton-line skeleton-label" /><span className="skeleton-line skeleton-value" /></div><span className="skeleton-line skeleton-range" /></div>)}
  </div>;
}

function Slider({ icon: Icon, label, hint, value, onCommit, accent = '' }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = event => onCommit(Number(event.currentTarget.value));
  return <div className="slider-row">
    <div className="slider-top"><span className={`slider-icon ${accent}`}><Icon size={15} /></span><div className="slider-label">{label}<span>{hint}</span></div><span className={`slider-value ${accent}`}>{draft}<small>%</small></span></div>
    <input aria-label={label} type="range" min="0" max="100" value={draft} onChange={event => setDraft(Number(event.target.value))} onPointerUp={commit} onKeyUp={commit} onBlur={commit} style={{ '--value': `${draft}%` }} />
  </div>;
}

function ScoreBar({ score, color }) {
  return <div className={`score-track ${color}`}><span style={{ width: `${Math.min(100, Math.max(4, score * 100))}%` }} /></div>;
}

function EventCard({ item, rank, weighted = false, dimensions = [], scoreOverride }) {
  const score = scoreValue(item, weighted, scoreOverride);
  const breakdown = item.score_breakdown || {};
  const dimensionBreakdowns = item.dimension_breakdowns || {};
  const categories = item.categories || [];
  const boosted = weighted && dimensions.some(dimension => (item.domain_weights?.[dimension] ?? 0) >= 7);
  return <article className={`event-card ${weighted ? 'weighted-card' : 'naive-card'} ${boosted ? 'boosted-card' : ''}`}>
    <div className="card-topline"><span className={`rank ${weighted ? 'rank-green' : ''}`}>{String(rank).padStart(2, '0')}</span><span className={`event-tag ${weighted && boosted ? 'tag-green' : ''}`}>{categories.slice(0, 2).join(' · ').toUpperCase() || 'MEMORY'}</span><span className="card-date">{formatDate(item.timestamp)}</span></div>
    <p className="event-copy">{item.text}</p>
    <div className="card-bottom"><div className="score-copy"><span>{weighted ? 'FINAL SCORE' : 'SIMILARITY'}</span><strong className={weighted ? 'green-text' : ''}>{score.toFixed(2)}</strong></div><ScoreBar score={weighted ? score / 5 : score} color={weighted ? 'bar-green' : 'bar-muted'} />{weighted && boosted && <span className="boost-pill"><Sparkles size={11} /> high impact</span>}</div>
    {weighted && <div className="score-breakdown"><span>SIM <b>{Number(breakdown.similarity ?? item.sim ?? 0).toFixed(2)}</b></span>{dimensions.map(dimension => <span key={dimension}>{dimension.split('_')[0].toUpperCase()} <b className="green-text">{Number(dimensionBreakdowns[dimension]?.contribution ?? 0).toFixed(2)}</b></span>)}<span>AGE <b>{item.age_days ?? '—'}d</b></span></div>}
  </article>;
}

function App() {
  const [query, setQuery] = useState(defaultQuery);
  const [submittedQuery, setSubmittedQuery] = useState(defaultQuery);
  const [domain, setDomain] = useState(projects[0].id);
  const [schema, setSchema] = useState(null);
  const [schemaState, setSchemaState] = useState('loading');
  const [weights, setWeights] = useState(initialWeights);
  const [dimensionBiases, setDimensionBiases] = useState({});
  const [selectedDimensions, setSelectedDimensions] = useState([]);
  const [dimensionMenuOpen, setDimensionMenuOpen] = useState(false);
  const [dateRange, setDateRange] = useState({ since: '', until: '' });
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [naiveResults, setNaiveResults] = useState([]);
  const [weightedResults, setWeightedResults] = useState([]);
  const [naiveState, setNaiveState] = useState('loading');
  const [weightedState, setWeightedState] = useState('loading');
  const [retryToken, setRetryToken] = useState(0);
  const [naiveError, setNaiveError] = useState('');
  const [weightedError, setWeightedError] = useState('');
  const [schemaError, setSchemaError] = useState('');
  const recallCacheRef = useRef(new Map());
  const recallContextRef = useRef('');
  const previousBiasesRef = useRef({});
  const previousActiveDimensionsRef = useRef([]);
  const update = (key, value) => setWeights(old => ({ ...old, [key]: value }));
  const updateDimensionBias = (key, value) => setDimensionBiases(old => ({ ...old, [key]: value }));
  const projectConfig = projects.find(item => item.id === domain) || projects[0];
  const schemaLoading = schemaState === 'loading' || Boolean(schema?._id && schema._id !== domain);
  const dimensions = useMemo(() => Object.keys(schema?.dimensions || {}).length ? Object.keys(schema.dimensions) : [projectConfig.dimension], [schema, projectConfig.dimension]);
  const activeDimensions = useMemo(() => dimensions.filter(dimension => selectedDimensions.includes(dimension)), [dimensions, selectedDimensions]);
  const categories = schema?.categories || projectConfig.categories;
  useEffect(() => {
    setSelectedDimensions(current => {
      const valid = current.filter(dimension => dimensions.includes(dimension));
      return valid.length ? valid : dimensions.slice(0, 1);
    });
  }, [dimensions]);
  const toggleDimension = dimension => setSelectedDimensions(current => {
    if (current.includes(dimension)) return current.length > 1 ? current.filter(item => item !== dimension) : current;
    return [...current, dimension];
  });
  const commonRequest = useMemo(() => ({ query: submittedQuery, project: domain, categories, since: toIso(dateRange.since), until: toIso(dateRange.until), reinforce: false }), [submittedQuery, domain, categories, dateRange]);
  const naiveRequest = useMemo(() => ({ query: commonRequest.query, project: commonRequest.project, categories: commonRequest.categories, since: commonRequest.since, until: commonRequest.until, limit: resultLimit }), [commonRequest]);
  const weightedRequests = useMemo(() => activeDimensions.map(dimension => ({
    ...commonRequest,
    dimension,
    limit: resultLimit,
    use_gates: true,
    w_vector: 1,
    w_domain_bias: (dimensionBiases[dimension] ?? 30) / 100 * 0.5,
    w_reinforce: 0.1,
  })), [commonRequest, activeDimensions, dimensionBiases]);
  const displayWeights = Object.fromEntries(activeDimensions.map(dimension => [dimension, dimensionBiases[dimension] ?? 30]));
  const pipeline = useMemo(() => JSON.stringify({
    endpoints: { raw_mongo_search: `${API_BASE_URL}/raw_recall`, weighted: `${API_BASE_URL}/recall` },
    raw_mongo_search: naiveRequest,
    pensieve_weighted_by_dimension: weightedRequests,
    note: 'The API embeds the query and runs its Atlas aggregation pipeline. Dimension result sets are merged using each dimension current_weight, which the scheduled decay task has already updated.'
  }, null, 2), [naiveRequest, weightedRequests]);

  useEffect(() => {
    const controller = new AbortController();
    setSchema(null);
    setSchemaState('loading');
    setSchemaError('');
    getProjectSchema(domain, controller.signal).then(value => {
      setSchema(value);
      setSchemaState('ready');
      setSchemaError('');
    }).catch(error => {
      if (error.name !== 'AbortError') {
        setSchemaState('error');
        setSchemaError(`Project schema failed: ${error.message}`);
      }
    });
    return () => controller.abort();
  }, [domain]);

  useEffect(() => {
    const controller = new AbortController();
    setNaiveState('loading');
    setNaiveResults([]);
    setNaiveError('');
    if (schemaLoading) return () => controller.abort();
    const timer = setTimeout(() => {
      rawRecall(naiveRequest, controller.signal).then(results => {
        setNaiveResults(results);
        setNaiveState('ready');
        setNaiveError('');
      }).catch(error => {
        if (error.name !== 'AbortError') {
          setNaiveResults(mockNaiveResults(domain, dateRange));
          setNaiveState('fallback');
          setNaiveError(`Raw recall failed: ${error.message} Showing mock fallback data.`);
        }
      });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [naiveRequest, retryToken, domain, dateRange, weights.similarity, schemaLoading]);

  useEffect(() => {
    const controller = new AbortController();
    setWeightedState('loading');
    setWeightedError('');
    if (schemaLoading) return () => controller.abort();
    const contextKey = JSON.stringify({ commonRequest, dimensions, retryToken });
    const contextChanged = recallContextRef.current !== contextKey;
    if (contextChanged) {
      recallContextRef.current = contextKey;
      recallCacheRef.current.clear();
    }
    const changedDimensions = contextChanged
      ? activeDimensions
      : activeDimensions.filter(dimension => !previousActiveDimensionsRef.current.includes(dimension) || (previousBiasesRef.current[dimension] ?? 30) !== (dimensionBiases[dimension] ?? 30));
    previousBiasesRef.current = { ...dimensionBiases };
    previousActiveDimensionsRef.current = [...activeDimensions];
    setWeightedResults([]);
    setWeightedState('loading');
    const requestsToFetch = changedDimensions.map(dimension => weightedRequests.find(request => request.dimension === dimension));
    const timer = setTimeout(() => {
      Promise.all(requestsToFetch.map(request => recall(request, controller.signal))).then(resultSets => {
        resultSets.forEach((results, index) => recallCacheRef.current.set(changedDimensions[index], results));
        const merged = new Map();
        activeDimensions.forEach(selectedDimension => {
          const results = recallCacheRef.current.get(selectedDimension) || [];
          results.forEach(item => {
            const id = item._id || item.id;
            if (!merged.has(id)) merged.set(id, { ...item, dimension_breakdowns: {} });
            const combined = merged.get(id);
            const ageDays = Number(item.age_days ?? Math.max(0, (Date.now() - new Date(item.timestamp).getTime()) / 86400000));
            combined.age_days = ageDays;
            for (const dimension of activeDimensions) {
              const weight = Number(item.current_weights?.[dimension] ?? (dimension === selectedDimension ? item.weight : undefined) ?? item.domain_weights?.[dimension] ?? schema?.dimensions?.[dimension]?.default ?? 0);
              const bias = (dimensionBiases[dimension] ?? 30) / 100 * 0.5;
              combined.dimension_breakdowns[dimension] = { weight, contribution: weight * bias };
            }
          });
        });
        const ranked = [...merged.values()].map(item => {
          const similarity = Number(item.sim ?? item.score_breakdown?.similarity ?? 0);
          const reinforcement = Math.log1p(Number(item.access_count ?? 0)) * 0.1;
          const domainScore = Object.values(item.dimension_breakdowns).reduce((sum, part) => sum + part.contribution, 0);
          return { ...item, final_score: similarity + domainScore + reinforcement, score_breakdown: { similarity, reinforcement, ...item.score_breakdown } };
        }).sort((a, b) => b.final_score - a.final_score).slice(0, resultLimit);
        setWeightedResults(ranked);
        setWeightedState('ready');
        setWeightedError('');
      }).catch(error => {
        if (error.name !== 'AbortError') {
          setWeightedResults(mockResults(domain, activeDimensions, schema, dimensionBiases, dateRange).weighted);
          setWeightedState('fallback');
          setWeightedError(`Weighted recall failed: ${error.message} Showing mock fallback data.`);
        }
      });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
      if (contextChanged && recallContextRef.current === contextKey) recallContextRef.current = '';
    };
  }, [weightedRequests, retryToken, activeDimensions, schema, schemaState, schemaLoading, dimensionBiases, dateRange, domain, commonRequest]);

  const submitSearch = () => {
    setSubmittedQuery(query.trim() || defaultQuery);
    setRetryToken(value => value + 1);
    setNaiveError('');
    setWeightedError('');
    setSchemaError('');
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#top"><span className="brand-mark"><Layers3 size={17} strokeWidth={2.2} /></span><span>pensieve<span className="brand-light">core</span></span></a>
      <div className="side-label">WORKSPACE</div>
      <button className="side-link active"><SlidersHorizontal size={16} /><span>Inspector</span><span className="side-active-dot" /></button>
      <button className="side-link" onClick={() => setPipelineOpen(!pipelineOpen)}><Database size={16} /><span>Memory store</span></button>
      <div className="side-label side-label-spaced">PROJECT DIMENSIONS</div>
      {dimensions.map((item, index) => <div className="dimension-item" key={item}><span className={`dim-ico ${index === 0 ? 'dim-finance' : index === 1 ? 'dim-time' : 'dim-emotion'}`}><Wallet size={14} /></span><span>{readable(item)}</span></div>)}
      <div className="sidebar-bottom"><div className="profile-avatar">A</div><div className="profile-copy"><strong>Alex Morgan</strong><span>Personal workspace</span></div><ChevronDown size={14} className="profile-chevron" /></div>
    </aside>

    <main id="top" className="main-content">
      <header className="topbar"><div className="breadcrumb">PLAYGROUND <span>/</span> <b>RETRIEVAL INSPECTOR</b></div><div className="topbar-right"><span className={`live-status ${naiveState === 'fallback' || weightedState === 'fallback' ? 'fallback-status' : ''}`}><i /> {naiveState === 'fallback' || weightedState === 'fallback' ? 'MOCK FALLBACK' : 'LIVE API'}</span><span className="topbar-divider" /><button className="help-button" aria-label="About Pensieve"><CircleHelp size={17} /></button><span className="key-hint"><Command size={11} /> K</span></div></header>
      <div className="content-wrap">
        <section className="page-heading"><div><div className="eyebrow"><span /> MEMORY RETRIEVAL PLAYGROUND</div><h1>Search your <em>life.</em></h1><p className="heading-sub">See what changes when your memories carry meaning.</p></div><div className="heading-note"><Sparkles size={14} /><span>Interactive demo</span></div></section>

        <section className="search-panel panel">
          <div className="panel-eyebrow"><span className="step-num">01</span><span>RETRIEVAL QUERY</span><span className="panel-eyebrow-line" /><span className="mock-chip">PENSIEVE API</span></div>
          <div className="query-row"><Search className="query-icon" size={19} /><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') submitSearch(); }} placeholder="Ask your memory anything..." aria-label="Search query" /><kbd>↵</kbd></div>
          <div className="search-footer"><label className="domain-label" htmlFor="domain-select">PROJECT</label><div className="select-wrap"><Landmark size={14} /><select id="domain-select" value={domain} onChange={e => setDomain(e.target.value)}>{projects.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select><ChevronDown size={13} /></div><span className="footer-divider" /><span className="memory-count"><Database size={13} /> {weightedState === 'loading' ? 'Loading results…' : `${weightedResults.length} ${weightedState === 'fallback' ? 'mock' : 'live'} hits`}</span><button className="search-again" onClick={submitSearch}>Search memories <ArrowRight size={14} /></button></div>
          <div className="time-window-row"><div className="time-window-title"><Clock3 size={14} /><span>TIME RANGE</span><small>Optional · same filter for both columns</small></div><label>FROM<input aria-label="Start date and time" type="datetime-local" value={dateRange.since} max={dateRange.until || undefined} onChange={event => setDateRange(old => ({ ...old, since: event.target.value }))} /></label><ArrowRight className="time-range-arrow" size={13} /><label>TO<input aria-label="End date and time" type="datetime-local" value={dateRange.until} min={dateRange.since || undefined} onChange={event => setDateRange(old => ({ ...old, until: event.target.value }))} /></label>{(dateRange.since || dateRange.until) && <button className="clear-range" onClick={() => setDateRange({ since: '', until: '' })}>Clear</button>}</div>
        </section>

        {(naiveError || weightedError || schemaError) && <div className="api-error" role="alert"><span><CircleHelp size={14} /> {[naiveError, weightedError, schemaError].filter(Boolean).join(' ')}</span><button onClick={submitSearch}>Retry</button></div>}

        <section className="results-section"><div className="results-heading"><div><div className="panel-eyebrow"><span className="step-num">03</span><span>RETRIEVAL COMPARISON</span></div><h2>Same query. <em>Live memories.</em></h2></div><span className="results-count"><Filter size={13} /> {weightedResults.length} WEIGHTED HITS</span></div>
          <div className="results-grid">
            <div className="lane-controls panel naive-controls"><div className="lane-controls-heading"><div><span>LEFT PANEL CONTROL</span><strong>Raw Mongo search</strong></div><span className="lane-scope">SIMILARITY ONLY</span></div><Slider icon={Search} label="Vector similarity" hint="Score display · refreshes raw results" value={weights.similarity} onCommit={v => update('similarity', v)} /></div>
            <div className="lane-controls panel weighted-controls"><div className="lane-controls-heading"><div><span>RIGHT PANEL CONTROLS</span><strong>Pensieve dimensions</strong></div><button className="reset-weights" onClick={() => { setWeights(initialWeights); setDimensionBiases({}); }}>Reset defaults <History size={12} /></button></div>{schemaLoading ? <DimensionSliderSkeletons /> : <><div className="dimension-selectors"><span className="dimension-select-label">QUERY DIMENSIONS</span><div className="dimension-select-wrap"><button className={`dimension-select-trigger ${dimensionMenuOpen ? 'open' : ''}`} type="button" aria-haspopup="listbox" aria-expanded={dimensionMenuOpen} onClick={() => setDimensionMenuOpen(open => !open)}><span>{activeDimensions.map(readable).join(', ')}</span><ChevronDown size={13} /></button>{dimensionMenuOpen && <div className="dimension-select-menu" role="group" aria-label="Dimensions to query">{dimensions.map(dimension => <label className="dimension-option" key={dimension}><input type="checkbox" checked={activeDimensions.includes(dimension)} disabled={activeDimensions.length === 1 && activeDimensions.includes(dimension)} onChange={() => toggleDimension(dimension)} /><span className="dimension-checkmark" /> <span>{readable(dimension)}</span></label>)}</div>}</div></div><div className="dimension-sliders">{activeDimensions.map(dimension => <Slider key={dimension} icon={Wallet} label={readable(dimension)} hint="Dimension importance · w_domain_bias" value={displayWeights[dimension]} onCommit={value => updateDimensionBias(dimension, value)} accent="green-accent" />)}</div></>}<div className="lane-control-note">Only selected dimensions are queried and scored.</div></div>
            <div className="result-column naive-column"><div className="column-head"><div className="column-icon naive-icon"><ArrowDownWideNarrow size={16} /></div><div className="column-title"><h3>Raw Mongo search</h3><span>{naiveState === 'fallback' ? 'MOCK FALLBACK' : 'RAW VECTOR SEARCH · API'}</span></div><span className="column-badge">BASELINE</span></div><div className="column-explainer">Ranked by semantic similarity alone · score display scaled locally</div><div className="cards-list">{naiveState === 'loading' ? <LoadingSkeletons /> : naiveResults.map((item, i) => <EventCard key={item._id} item={item} rank={i + 1} scoreOverride={Number(item.similarity ?? item.sim ?? item.score_breakdown?.similarity ?? 0) * weights.similarity / 100} />)}{['ready', 'fallback'].includes(naiveState) && !naiveResults.length && <div className="result-message">No matching memories in this project.</div>}</div><div className="column-foot"><span className="foot-dot orange-dot" /> Same project and query, similarity only.</div></div>
            <div className="result-column weighted-column"><div className="column-head"><div className="column-icon weighted-icon"><Sparkles size={16} /></div><div className="column-title"><h3>Pensieve retrieval</h3><span>{weightedState === 'fallback' ? 'MOCK FALLBACK' : 'WEIGHTED RECALL · API'}</span></div><span className="column-badge green-badge">CONTEXT AWARE</span></div><div className="column-explainer">Weighted by {activeDimensions.length} selected dimension{activeDimensions.length === 1 ? '' : 's'}<span className="live-chip"><i /> {weightedState === 'loading' ? 'UPDATING' : weightedState === 'fallback' ? 'FALLBACK' : 'LIVE'}</span></div><div className="cards-list">{weightedState === 'loading' ? <LoadingSkeletons weighted /> : weightedResults.map((item, i) => <EventCard key={item._id} item={item} rank={i + 1} weighted dimensions={activeDimensions} />)}{['ready', 'fallback'].includes(weightedState) && !weightedResults.length && <div className="result-message">No matching memories in this project.</div>}</div><div className="column-foot green-foot"><Sparkles size={13} /> {weightedState === 'fallback' ? 'Using local demo scores because the API request failed.' : 'Scores and rankings come from Pensieve Core.'}</div></div>
          </div>
        </section>

        <section className={`pipeline-panel panel ${pipelineOpen ? 'pipeline-expanded' : ''}`}><button className="pipeline-toggle" onClick={() => setPipelineOpen(!pipelineOpen)} aria-expanded={pipelineOpen}><div className="pipeline-title-icon"><Braces size={15} /></div><div className="pipeline-heading"><strong>Under the hood</strong><span>Live requests sent to Pensieve Core</span></div><span className="pipeline-live"><i /> {API_BASE_URL.replace('https://', '')}</span><ChevronDown size={15} className="pipeline-chevron" /></button>{pipelineOpen && <div className="pipeline-code"><div className="code-toolbar"><span><FileCode2 size={13} /> recall.requests.json</span><span>LIVE API · EMBEDDINGS OMITTED</span></div><pre>{pipeline}</pre></div>}</section>
        <footer className="page-footer"><span><span className="brand-mini">P</span> PENSIEVE CORE <span className="footer-version">LIVE API · {projectConfig.id}</span></span><span>MEANING, NOT JUST MATCHING <Sparkles size={12} /></span></footer>
      </div>
    </main>
  </div>;
}

export default App;
