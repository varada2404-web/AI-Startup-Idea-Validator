const API_BASE = window.VALIDATOR_API_BASE || window.location.origin;
const form = document.querySelector("#idea-form");
const ideaInput = document.querySelector("#idea");
const charCount = document.querySelector("#char-count");
const submitButton = document.querySelector("#submit-button");
const progressMessage = document.querySelector("#progress-message");
const progressDot = document.querySelector(".progress-summary .status-dot");
const stageList = document.querySelector("#stage-list");
const runError = document.querySelector("#run-error");
const runIdElement = document.querySelector("#run-id");
const resultsSection = document.querySelector("#results");
const ACTIVE_RUN_KEY = "ideasignal.activeRunId";

const CATEGORY_LABELS = {
  customer_pain: "Customer pain",
  market: "Market context",
  direct_competitors: "Direct competitors",
  alternatives: "Alternatives",
  pricing: "Pricing",
  recent_developments: "Recent developments",
  negative_evidence: "Negative evidence",
};

ideaInput.addEventListener("input", () => {
  charCount.textContent = `${ideaInput.value.length.toLocaleString()} / 1,000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  runError.hidden = true;
  resultsSection.hidden = true;
  setBusy(true);
  setOverallState("running", "Submitting your idea to the validation service…");
  document.querySelector("#run-status").textContent = "Submitting";
  document.querySelector("#pipeline-empty").hidden = true;
  stageList.replaceChildren();
  resetStages();

  const payload = {
    idea: ideaInput.value,
    geography: document.querySelector("#geography").value.trim() || null,
    target_customer: document.querySelector("#target-customer").value.trim() || null,
  };

  try {
    const response = await fetchApi(`${API_BASE}/api/v1/validations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const accepted = await readResponse(response);
    sessionStorage.setItem(ACTIVE_RUN_KEY, accepted.run_id);
    setApplicationMode(accepted.mode);
    runIdElement.textContent = `Run ${accepted.run_id}`;
    runIdElement.hidden = false;
    document.querySelector("#context-run-id").textContent = accepted.run_id;
    document.querySelector("#context-idea").textContent = payload.idea;
    document.querySelector("#idea-card").classList.add("has-submission");
    await pollRun(accepted.run_id);
  } catch (error) {
    setOverallState("failed", "The request could not be completed.");
    document.querySelector("#run-status").textContent = "Failed";
    if (error instanceof ConnectivityError) {
      showConnectionError();
    } else {
      showError(error instanceof ApiResponseError
        ? error.message
        : "Something went wrong while building your research brief. Please try again.");
    }
    setBusy(false);
  }
});

class ApiResponseError extends Error {}
class ConnectivityError extends Error {}

async function fetchApi(url, options) {
  try {
    return await fetch(url, options);
  } catch (error) {
    if (error instanceof TypeError) throw new ConnectivityError();
    throw error;
  }
}

async function readResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new ApiResponseError(body?.error?.message || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return body;
}

async function pollRun(runId) {
  const startedAt = Date.now();
  let continuationShown = false;
  while (true) {
    try {
      const response = await fetchApi(`${API_BASE}/api/v1/validations/${encodeURIComponent(runId)}`);
      const run = await readResponse(response);
      renderProgress(run);
      document.querySelector("#context-run-id").textContent = run.run_id || runId;
      document.querySelector("#context-idea").textContent = run.request?.idea || "";
      runIdElement.textContent = `Run ${run.run_id || runId}`;
      runIdElement.hidden = false;
      if (run.status === "completed") {
        sessionStorage.removeItem(ACTIVE_RUN_KEY);
        renderResults(run);
        const completionMessage = run.quality === "degraded"
          ? "Research completed with degraded coverage. Review the disclosure below."
          : run.simulation_notice || "Research brief ready.";
        setOverallState("done", completionMessage);
        setBusy(false);
        return;
      }
      if (["failed", "interrupted"].includes(run.status)) {
        sessionStorage.removeItem(ACTIVE_RUN_KEY);
        setOverallState("failed", run.error || `The run was ${run.status}.`);
        document.querySelector("#run-status").textContent = formatLabel(run.status);
        showError(run.error || `The run was ${run.status}. You can submit another idea.`);
        setBusy(false);
        return;
      }
    } catch (error) {
      const transient = error instanceof ConnectivityError
        || (error instanceof ApiResponseError && error.status >= 500);
      if (!transient) {
        sessionStorage.removeItem(ACTIVE_RUN_KEY);
        showError(error instanceof ApiResponseError ? error.message : "Unable to read this run.");
        setBusy(false);
        return;
      }
      setOverallState("running", "Connection interrupted. Reconnecting to the active run…");
    }
    if (!continuationShown && Date.now() - startedAt >= 36_000) {
      continuationShown = true;
      document.querySelector("#run-status").textContent = "Still running";
      setOverallState("running", `This run is taking longer than usual. We’ll keep checking run ${runId}; the form stays locked to prevent a duplicate submission.`);
    }
    await delay(1_000);
  }
}

function renderProgress(run) {
  setApplicationMode(run.mode);
  document.querySelector("#run-status").textContent = formatLabel(run.status);
  const stages = Array.isArray(run.progress) ? run.progress : [];
  stageList.replaceChildren(...stages.map((stage, index) => {
    const row = element("li", `pipeline-stage status-${safeStatus(stage.status)}`);
    row.dataset.status = safeStatus(stage.status);
    const indexNode = element("span", "stage-index", String(index + 1).padStart(2, "0"));
    const content = element("span", "stage-content");
    content.append(element("span", "stage-label", stage.label));
    if (stage.detail) content.append(element("span", "stage-detail", stage.detail));
    row.append(indexNode, content, element("span", "stage-state", formatLabel(stage.status)));
    return row;
  }));
  document.querySelector("#pipeline-empty").hidden = stages.length > 0;
  const active = stages.find((stage) => stage.status === "running");
  if (active) setOverallState("running", active.detail || active.label);
}

function renderResults(run) {
  setApplicationMode(run.mode);
  const brief = run.idea_brief;
  if (brief) {
    renderBrief(brief);
    renderHypotheses(Array.isArray(brief.hypotheses) ? brief.hypotheses : []);
  } else {
    document.querySelector("#brief-content").replaceChildren(element("p", "empty-note", "No idea brief was returned for this run."));
    document.querySelector("#hypotheses-content").replaceChildren();
  }
  renderQueries(Array.isArray(run.queries) ? run.queries : []);
  renderCoverage(Array.isArray(run.coverage) ? run.coverage : []);
  renderSources(Array.isArray(run.sources) ? run.sources : []);

  const notice = document.querySelector("#simulation-notice");
  const providerMessages = (run.provider_errors || []).map((item) => item.message).filter(Boolean);
  const qualityNotice = run.quality === "degraded"
    ? "DEGRADED RESEARCH: some provider requests failed or source pages could not be extracted. Successful results and available snippets are retained."
    : null;
  notice.textContent = [run.simulation_notice, qualityNotice, ...providerMessages].filter(Boolean).join(" ");
  notice.hidden = !notice.textContent;
  document.querySelector("#context-run-id").textContent = run.run_id || "";
  document.querySelector("#context-idea").textContent = run.request?.idea || "";
  resultsSection.hidden = false;
  document.querySelector("#nav-results-link").hidden = false;
  document.querySelector("#rail-results-link").hidden = false;
  document.querySelector("#results-title").focus({ preventScroll: true });
  resultsSection.scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    block: "start",
  });
}

function renderBrief(brief) {
  const container = document.querySelector("#brief-content");
  const lines = [
    ["Original wording", brief.original_text],
    ["Problem interpretation", brief.problem],
    ["Solution", brief.solution],
    ["Target customer", brief.target_customer?.value, brief.target_customer?.origin],
    ["Geography", brief.geography?.value, brief.geography?.origin],
    ["Industry", brief.industry],
    ["Business model", brief.business_model],
  ];
  const rendered = lines
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value, provenance]) => briefLine(label, String(value), provenance));

  if (Array.isArray(brief.missing_info) && brief.missing_info.length) {
    const missing = element("div", "missing-info");
    missing.append(element("span", "brief-label", "Open questions"));
    const list = element("ul", "missing-list");
    list.append(...brief.missing_info.map((item) => element("li", "", item)));
    missing.append(list);
    rendered.push(missing);
  }

  if (Array.isArray(brief.search_keywords) && brief.search_keywords.length) {
    const keywords = element("div", "keyword-row");
    keywords.append(element("span", "brief-label", "Search keywords"));
    const list = element("div", "keyword-list");
    list.append(...brief.search_keywords.map((keyword) => element("span", "keyword-chip", keyword)));
    keywords.append(list);
    rendered.push(keywords);
  }
  container.replaceChildren(...rendered);
}

function briefLine(label, value, provenance = "") {
  const wrapper = element("div", "brief-line");
  wrapper.append(element("span", "brief-label", label));
  const content = element("p", "brief-value", value);
  if (provenance) content.append(element("span", "provenance", formatLabel(provenance)));
  wrapper.append(content);
  return wrapper;
}

function renderHypotheses(hypotheses) {
  const container = document.querySelector("#hypotheses-content");
  container.replaceChildren(...hypotheses.map((hypothesis, index) => {
    const card = element("article", "hypothesis-card reveal");
    const indexNode = element("span", "hypothesis-id", hypothesis.id);
    const body = element("div", "hypothesis-body");
    const title = element("div", "hypothesis-kind", formatLabel(hypothesis.kind));
    if (hypothesis.critical) title.append(element("span", "critical-tag", "Default critical"));
    const status = safeStatus(hypothesis.status);
    body.append(title, element("p", "hypothesis-statement", hypothesis.statement));
    body.append(element("span", `hypothesis-status status-${status}`, formatLabel(hypothesis.status)));
    card.append(indexNode, body);
    return card;
  }));
  observeReveals(container);
}

function renderCoverage(items) {
  const container = document.querySelector("#coverage-grid");
  container.replaceChildren(...items.map((item, index) => {
    const card = element("article", "coverage-card reveal");
    card.append(element("span", "coverage-index", String(index + 1).padStart(2, "0")));
    card.append(element("h4", "coverage-name", CATEGORY_LABELS[item.category] || formatLabel(item.category)));
    card.append(element("span", `coverage-status status-${safeStatus(item.status)}`, formatLabel(item.status)));
    const facts = element("div", "coverage-facts");
    facts.append(
      coverageFact("Sources", item.source_count),
      coverageFact("Independent clusters", item.independent_clusters),
    );
    card.append(facts);
    return card;
  }));
  if (!items.length) container.append(element("p", "empty-note", "No category coverage was returned for this run."));
  observeReveals(container);
}

function coverageFact(label, value) {
  const fact = element("span", "coverage-fact");
  fact.append(element("span", "coverage-fact-value", String(value)), element("span", "coverage-fact-label", label));
  return fact;
}

function renderQueries(queries) {
  const container = document.querySelector("#query-list");
  document.querySelector("#query-count").textContent = `${queries.length} planned`;
  container.replaceChildren(...queries.map((query) => {
    const row = element("li", "query-item");
    row.append(
      element("span", "query-category", CATEGORY_LABELS[query.category] || formatLabel(query.category)),
      element("span", "query-round", `Round ${query.round}`),
      element("span", "query-text", query.query),
    );
    return row;
  }));
}

function renderSources(sources) {
  const container = document.querySelector("#source-groups");
  const groups = new Map();
  for (const source of sources) {
    for (const category of source.categories || []) {
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(source);
    }
  }
  document.querySelector("#source-count").textContent = `${sources.length} unique source records`;
  container.replaceChildren(...[...groups.entries()].map(([category, records]) => {
    const group = element("section", "source-group");
    group.append(element("h4", "source-group-title", CATEGORY_LABELS[category] || formatLabel(category)));
    for (const source of records) group.append(sourceCard(source));
    return group;
  }));
  if (!groups.size) container.append(element("p", "empty-note", "No source records were returned for this run."));
}

function sourceCard(source) {
  const card = element("article", "source-card");
  const heading = element("div", "source-card-heading");
  const title = element("h5", "source-title", source.title);
  heading.append(title);
  if (Array.isArray(source.queries) && source.queries.length > 1) {
    heading.append(element("span", "source-merged", "Merged duplicate"));
  }
  card.append(heading);

  const meta = [source.domain, source.source_type, source.simulated ? "simulated" : ""]
    .filter(Boolean).map(formatLabel).join(" · ");
  card.append(element("p", "source-meta", meta));

  const details = document.createElement("details");
  details.className = "source-details";
  const summary = document.createElement("summary");
  summary.append(element("span", "", "Inspect source record"), element("span", "details-icon", "+"));
  details.append(summary);
  const body = element("div", "source-detail-body");
  const snippet = source.snippet ?? source.text;
  if (snippet) {
    body.append(element("span", "brief-label", "Search snippet"));
    body.append(element("p", "source-text", snippet));
  }
  body.append(element("span", "brief-label", `Extraction ${formatLabel(source.extraction_status || "not_attempted")}`));
  if (source.extracted_text) {
    body.append(element("span", "brief-label", "Extracted page content · not independently verified"));
    body.append(element("p", "source-text", source.extracted_text));
  }
  if (source.extraction_error?.message) {
    body.append(element("p", "empty-note", source.extraction_error.message));
  }
  const metadata = element("dl", "source-provenance");
  appendDefinition(metadata, "Source URL", source.url, true);
  appendDefinition(metadata, "Canonical URL", source.canonical_url, true);
  appendDefinition(metadata, "Original query", source.query);
  if (Array.isArray(source.queries) && source.queries.length) appendDefinition(metadata, "Matched queries", source.queries.join(" · "));
  appendDefinition(metadata, "Cluster", source.cluster_id);
  appendDefinition(metadata, "Retrieved", source.retrieved_at ? new Date(source.retrieved_at).toLocaleString() : "");
  appendDefinition(metadata, "Published", source.published_at ? new Date(source.published_at).toLocaleString() : "");
  appendDefinition(metadata, "Provider score", source.provider_score);
  body.append(metadata);
  details.append(body);
  details.addEventListener("toggle", () => animateDisclosure(details));
  card.append(details);
  return card;
}

function appendDefinition(list, label, value, link = false) {
  if (value === null || value === undefined || value === "") return;
  list.append(element("dt", "", label));
  const description = element("dd");
  if (link) {
    const anchor = element("a", "source-url", value);
    anchor.href = value;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    description.append(anchor);
  } else {
    description.textContent = value;
  }
  list.append(description);
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function setOverallState(state, message) {
  progressMessage.textContent = message;
  progressDot.className = `status-dot status-${safeStatus(state)}`;
}

function showError(message) {
  runError.textContent = message;
  runError.hidden = false;
}

function showConnectionError() {
  const backendLink = element("a", "", "http://127.0.0.1:8000");
  backendLink.href = "http://127.0.0.1:8000";
  runError.replaceChildren(
    document.createTextNode("Unable to connect to the validation server. Please make sure the backend is running at "),
    backendLink,
    document.createTextNode(" and try again."),
  );
  runError.hidden = false;
}

function setBusy(busy) {
  submitButton.disabled = busy;
  submitButton.firstElementChild.textContent = busy ? "Building your brief…" : "Build my research brief";
  submitButton.setAttribute("aria-busy", String(busy));
}

function resetStages() {
  stageList.querySelectorAll(".pipeline-stage").forEach((stage) => {
    stage.classList.remove("is-running", "is-completed", "is-failed");
  });
}

function safeStatus(value) {
  const allowed = new Set(["queued", "running", "completed", "done", "failed", "interrupted", "pending", "unknown", "supported", "mixed", "contradicted", "covered", "partial", "missing"]);
  return allowed.has(value) ? value : "unknown";
}

function formatLabel(value) {
  return String(value || "Unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function observeReveals(root = document) {
  if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  root.querySelectorAll(".reveal, .section-reveal").forEach((item) => revealObserver.observe(item));
}

const revealObserver = "IntersectionObserver" in window
  ? new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 })
  : null;

observeReveals();
document.querySelectorAll(".query-details").forEach((details) => {
  details.addEventListener("toggle", () => animateDisclosure(details));
});

function animateDisclosure(details) {
  if (!details.open || reducedMotion.matches || typeof details.animate !== "function") return;
  const body = details.querySelector(".source-detail-body, .query-list");
  body?.animate(
    [{ opacity: 0, transform: "translateY(-5px)" }, { opacity: 1, transform: "translateY(0)" }],
    { duration: 190, easing: "ease-out" },
  );
}

const heroSection = document.querySelector(".hero");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let parallaxFrame = null;
if (heroSection && !reducedMotion.matches) {
  window.addEventListener("scroll", () => {
    if (parallaxFrame !== null) return;
    parallaxFrame = window.requestAnimationFrame(() => {
      if (reducedMotion.matches) {
        parallaxFrame = null;
        return;
      }
      const bounds = heroSection.getBoundingClientRect();
      if (bounds.bottom > 0 && bounds.top < window.innerHeight) {
        const drift = Math.max(-16, Math.min(16, (window.innerHeight - bounds.top) * 0.018 - 9));
        heroSection.style.setProperty("--scroll-drift", `${drift}px`);
      }
      parallaxFrame = null;
    });
  }, { passive: true });
  reducedMotion.addEventListener("change", (event) => {
    if (event.matches) heroSection.style.removeProperty("--scroll-drift");
  });
}

const rail = document.querySelector("#chapter-rail");
if (rail) {
  rail.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      scrollCarousel(event.key === "ArrowRight" ? 260 : -260);
    }
  });
  document.querySelector("#rail-prev").addEventListener("click", () => scrollCarousel(-260));
  document.querySelector("#rail-next").addEventListener("click", () => scrollCarousel(260));
}

function scrollCarousel(distance) {
  rail.scrollBy({ left: distance, behavior: reducedMotion.matches ? "auto" : "smooth" });
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function setApplicationMode(mode) {
  const isDemo = mode === "demo";
  const badgeText = isDemo ? "DEMO MODE" : "LIVE RESEARCH";
  for (const id of ["mode-badge", "results-mode-badge"]) {
    const badge = document.querySelector(`#${id}`);
    if (!badge) continue;
    const dot = badge.querySelector(".pulse-dot");
    badge.replaceChildren(...(dot ? [dot, document.createTextNode(` ${badgeText}`)] : [document.createTextNode(badgeText)]));
  }
  const modeNotice = document.querySelector("#mode-notice");
  const sourceCardCaption = document.querySelector("#hero-source-caption");
  if (sourceCardCaption) {
    sourceCardCaption.textContent = isDemo
      ? "Simulated placeholder · Not a finding"
      : "Retrieved page · Not verified";
  }
  const lead = element("strong", "", isDemo ? "Offline demo. " : "Live research. ");
  modeNotice.replaceChildren(
    lead,
    document.createTextNode(isDemo
      ? "Research records are deterministic synthetic placeholders. No live search or market research is performed."
      : "Gemini interprets the idea and Tavily retrieves public pages. Retrieval and extraction do not verify claims."),
  );
  document.querySelector("#privacy-note").textContent = isDemo
    ? "Your idea is sent only to this application. No external providers are used in DEMO mode."
    : "Your idea is sent to Gemini for interpretation; research queries and selected public URLs are sent to Tavily.";
  document.querySelector("#sources-eyebrow").textContent = isDemo
    ? "SIMULATED PLACEHOLDERS · NO LIVE SOURCES"
    : "LIVE RETRIEVAL · NOT INDEPENDENTLY VERIFIED";
  document.querySelector("#source-explainer").textContent = isDemo
    ? "These deterministic records demonstrate categorization and deduplication only. They are not real-world findings. Long source text is collapsed until opened."
    : "Search snippets and extracted page text are provider-returned material, not verified evidence or findings. Long text is collapsed until opened.";
  document.querySelector("#results-footer-copy").textContent = isDemo
    ? "M1 covers Agent 1: Web Search and Idea Understanding. These simulated records are not market findings. Agents 2–6 and any verdict are outside this prototype."
    : "M1 covers Agent 1: Web Search and Idea Understanding. Retrieved pages are not verified findings. Agents 2–6 and any verdict are outside this prototype.";
  document.querySelector("#page-footer-copy").textContent = isDemo
    ? "Agent 1 only · Synthetic demo · No market conclusions"
    : "Agent 1 only · Live retrieval · No claim verification or market conclusions";
}

function resumeActiveRun() {
  const runId = sessionStorage.getItem(ACTIVE_RUN_KEY);
  if (!runId) return;
  setBusy(true);
  runIdElement.textContent = `Run ${runId}`;
  runIdElement.hidden = false;
  document.querySelector("#context-run-id").textContent = runId;
  document.querySelector("#run-status").textContent = "Reconnecting";
  setOverallState("running", `Reconnecting to active run ${runId}…`);
  document.querySelector("#pipeline-empty").hidden = true;
  pollRun(runId).catch(() => {
    // pollRun handles transient failures internally; this is a final defensive guard.
    setOverallState("running", `Run ${runId} is still active. Keep this page open while status reconnects.`);
  });
}

async function loadApplicationMode() {
  try {
    const response = await fetchApi(`${API_BASE}/api/v1/health`);
    const health = await readResponse(response);
    setApplicationMode(String(health.mode || "").toLowerCase());
  } catch {
    document.querySelector("#mode-badge").lastChild.textContent = " MODE UNKNOWN";
    document.querySelector("#mode-notice").textContent = "Connect to the validation server to confirm provider mode.";
  }
}

loadApplicationMode();
resumeActiveRun();
