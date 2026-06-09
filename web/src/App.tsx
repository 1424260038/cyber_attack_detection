import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle,
  Database,
  Download,
  FileText,
  Network,
  Radar,
  Shield,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import "./App.css";

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: {
    Normal: number;
    Attack: number;
  };
  class_probabilities?: Record<string, number>;
  feature_contributions?: Array<{
    feature: string;
    index: number;
    value: number;
    score: number;
    direction: string;
  }>;
  attack_type?: string;
  risk_level?: string;
  recommendation?: string;
  attack_chain_cn?: string[];
  detected_techniques?: string[];
  chain_details?: Array<{
    tactic: string;
    tactic_cn: string;
    techniques: Array<{ id: string; name: string; name_cn: string }>;
  }>;
}

interface ApiMetadata {
  input_dim: number;
  num_classes: number;
  class_names: string[];
  architecture: string;
  checkpoint_path: string;
  best_val_acc?: number;
  preprocessor_loaded?: boolean;
  feature_columns?: string[];
}

interface MetricsPayload {
  test_results?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
    average_precision: number;
  } | null;
  training_history?: {
    train_loss?: number[];
    train_acc?: number[];
    val_loss?: number[];
    val_acc?: number[];
  } | null;
}

interface KnowledgeGraphSummary {
  entity_count: number;
  relation_count: number;
  tactics: Array<{ id: string; name: string; name_cn: string }>;
}

interface CsvDetectionResult {
  total_rows: number;
  processed_rows: number;
  used_columns: string[];
  summary: Record<string, number>;
  results: PredictionResult[];
  source_file?: string;
}

interface DetectionEvent {
  id: string;
  timestamp: string;
  source: string;
  prediction: string;
  confidence: number;
  attack_type?: string;
  risk_level?: string;
}

interface EventSummary {
  total_events: number;
  prediction_counts: Record<string, number>;
  risk_counts: Record<string, number>;
  attack_type_counts: Record<string, number>;
  latest?: DetectionEvent | null;
}

interface DatasetSummary {
  input?: string | null;
  files: string[];
  output?: string | null;
  rows: number;
  feature_columns: number;
  label_distribution: Record<string, number>;
}

interface DatasetProfile {
  rows: number;
  columns: number;
  numeric_feature_columns: number;
  missing_rate: number;
  class_count: number;
  imbalance_ratio: number;
  low_variance_columns: string[];
  warnings: string[];
  quality_score: number;
}

interface RuntimeArtifact {
  path: string;
  exists: boolean;
  size: number;
}

interface RuntimeStatus {
  model_loaded: boolean;
  preprocessor_loaded: boolean;
  knowledge_graph_loaded: boolean;
  artifacts: Record<string, RuntimeArtifact>;
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.DEV ? "http://localhost:8000" : window.location.origin);

const formatPercent = (value?: number) => (value === undefined ? "N/A" : `${(value * 100).toFixed(2)}%`);
const formatModelPercent = (value?: number) => (value === undefined ? "N/A" : `${value.toFixed(2)}%`);

const metricLabelMap: Record<string, string> = {
  accuracy: "准确率",
  precision: "精确率",
  recall: "召回率",
  f1_score: "F1",
  auc_roc: "AUC",
  average_precision: "AP",
};

function App() {
  const [features, setFeatures] = useState("");
  const [csvText, setCsvText] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [csvResult, setCsvResult] = useState<CsvDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState<"unknown" | "online" | "offline">("unknown");
  const [metadata, setMetadata] = useState<ApiMetadata | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraphSummary | null>(null);
  const [eventSummary, setEventSummary] = useState<EventSummary | null>(null);
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [datasetProfile, setDatasetProfile] = useState<DatasetProfile | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayResult, setReplayResult] = useState<CsvDetectionResult | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [reloadLoading, setReloadLoading] = useState(false);

  const expectedFeatureCount = metadata?.input_dim ?? 64;

  const latestValAcc = useMemo(() => {
    const history = metrics?.training_history;
    if (!history?.val_acc?.length) return undefined;
    return history.val_acc[history.val_acc.length - 1];
  }, [metrics]);

  const checkApiStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      const health = await res.json();
      if (!res.ok || !health.model_loaded) {
        setApiStatus("offline");
        return;
      }

      const [metadataRes, metricsRes, kgRes, eventSummaryRes, eventsRes, datasetSummaryRes, datasetProfileRes, runtimeRes] = await Promise.all([
        fetch(`${API_BASE_URL}/metadata`),
        fetch(`${API_BASE_URL}/metrics`),
        fetch(`${API_BASE_URL}/knowledge-graph`),
        fetch(`${API_BASE_URL}/events/summary`),
        fetch(`${API_BASE_URL}/events?limit=8`),
        fetch(`${API_BASE_URL}/dataset/summary`),
        fetch(`${API_BASE_URL}/dataset/profile`),
        fetch(`${API_BASE_URL}/admin/runtime`),
      ]);

      if (metadataRes.ok) setMetadata(await metadataRes.json());
      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (kgRes.ok) setKnowledgeGraph(await kgRes.json());
      if (eventSummaryRes.ok) setEventSummary(await eventSummaryRes.json());
      if (eventsRes.ok) setEvents((await eventsRes.json()).events);
      if (datasetSummaryRes.ok) setDatasetSummary(await datasetSummaryRes.json());
      if (datasetProfileRes.ok) setDatasetProfile(await datasetProfileRes.json());
      if (runtimeRes.ok) setRuntimeStatus(await runtimeRes.json());
      setApiStatus("online");
    } catch {
      setApiStatus("offline");
    }
  };

  useEffect(() => {
    checkApiStatus();
  }, []);

  const parseFeatures = () =>
    features
      .split(/[\s,]+/)
      .filter(Boolean)
      .map((f) => Number(f.trim()));

  const handlePredict = async () => {
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const featureArray = parseFeatures();
      if (featureArray.length === 0 || featureArray.some(Number.isNaN)) {
        throw new Error("请输入有效的数字，用逗号或空格分隔");
      }
      if (featureArray.length !== expectedFeatureCount) {
        throw new Error(`请输入${expectedFeatureCount}个特征，当前输入了${featureArray.length}个`);
      }

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: featureArray }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? "API请求失败");
      }

      setResult(await res.json());
      await refreshEvents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法连接到后端服务，请确保API服务已启动");
    } finally {
      setLoading(false);
    }
  };

  const loadDemoSample = async (kind: "normal" | "attack") => {
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/demo-samples`);
      if (!res.ok) throw new Error("无法加载样例，请确认 API 已启动");
      const data = await res.json();
      setFeatures(data[kind].join(", "));
    } catch (err) {
      setError(err instanceof Error ? err.message : "样例加载失败");
    }
  };

  const generateRandomFeatures = () => {
    const generated = Array.from({ length: expectedFeatureCount }, () => (Math.random() * 2 - 1).toFixed(4));
    setFeatures(generated.join(", "));
  };

  const generateDemoCsv = () => {
    const headers = Array.from({ length: expectedFeatureCount }, (_, index) => `f${index}`).join(",");
    const normal = Array.from({ length: expectedFeatureCount }, () => "0.05").join(",");
    const attack = Array.from({ length: expectedFeatureCount }, () => "2.00").join(",");
    setCsvText(`${headers}\n${normal}\n${attack}`);
  };

  const handleCsvPredict = async () => {
    setError("");
    setCsvResult(null);
    setCsvLoading(true);

    try {
      let res: Response;
      if (csvFile) {
        const formData = new FormData();
        formData.append("file", csvFile);
        formData.append("max_rows", "200");
        res = await fetch(`${API_BASE_URL}/predict/upload`, {
          method: "POST",
          body: formData,
        });
      } else {
        res = await fetch(`${API_BASE_URL}/predict/csv`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csv_text: csvText, max_rows: 200 }),
        });
      }

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? "CSV检测失败");
      }

      setCsvResult(await res.json());
      await refreshEvents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "CSV检测失败");
    } finally {
      setCsvLoading(false);
    }
  };

  const exportCsvResult = async () => {
    setError("");
    try {
      let res: Response;
      if (csvFile) {
        const formData = new FormData();
        formData.append("file", csvFile);
        formData.append("max_rows", "200");
        res = await fetch(`${API_BASE_URL}/predict/upload/export`, {
          method: "POST",
          body: formData,
        });
      } else {
        res = await fetch(`${API_BASE_URL}/predict/csv/export`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csv_text: csvText, max_rows: 200 }),
        });
      }

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? "导出批量结果失败");
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "prediction_results.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      await refreshEvents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "导出批量结果失败");
    }
  };

  const refreshEvents = async () => {
    const [summaryRes, eventsRes] = await Promise.all([
      fetch(`${API_BASE_URL}/events/summary`),
      fetch(`${API_BASE_URL}/events?limit=8`),
    ]);
    if (summaryRes.ok) setEventSummary(await summaryRes.json());
    if (eventsRes.ok) setEvents((await eventsRes.json()).events);
  };

  const clearEvents = async () => {
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/events`, { method: "DELETE" });
      if (!res.ok) throw new Error("清空事件失败");
      await refreshEvents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "清空事件失败");
    }
  };

  const exportEvents = () => {
    window.location.href = `${API_BASE_URL}/events/export.csv?limit=500`;
  };

  const replayDemoTraffic = async () => {
    setError("");
    setReplayLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/demo/replay?max_rows=16`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? "演示流量回放失败");
      }
      setReplayResult(await res.json());
      await refreshEvents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "演示流量回放失败");
    } finally {
      setReplayLoading(false);
    }
  };

  const reloadRuntime = async () => {
    setError("");
    setReloadLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/admin/reload`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? "运行时重载失败");
      }
      await checkApiStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行时重载失败");
    } finally {
      setReloadLoading(false);
    }
  };

  const openArtifact = (path: string) => {
    window.open(`${API_BASE_URL}${path}`, "_blank", "noopener,noreferrer");
  };

  const statusClass =
    apiStatus === "online"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-400/30"
      : apiStatus === "offline"
        ? "bg-red-500/15 text-red-300 border-red-400/30"
        : "bg-slate-500/15 text-slate-300 border-slate-400/30";

  return (
    <div className="cyber-shell min-h-screen text-slate-100">
      <div className="cyber-backdrop" />
      <header className="cyber-header">
        <div className="cyber-header-inner">
          <div className="flex items-center gap-3">
            <div className="brand-mark">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <div className="brand-title">CyberDD 控制台</div>
              <div className="brand-subtitle">网络攻击检测 · 知识图谱解释 · 演示交付</div>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
            <span className="nav-chip">模型推理</span>
            <span className="nav-chip">批量检测</span>
            <span className="nav-chip">运行审计</span>
            <span className="nav-chip">交付物</span>
          </div>
          <button onClick={checkApiStatus} className={`status-pill ${statusClass}`}>
            <span className="status-dot" />
            {apiStatus === "online" ? "API 在线" : apiStatus === "offline" ? "API 离线" : "检测中"}
          </button>
        </div>
      </header>

      <main className="cyber-main">
        <section className="hero-grid mb-6">
          <div className="hero-card">
            <div className="hero-orbit hero-orbit-one" />
            <div className="hero-orbit hero-orbit-two" />
            <div className="hero-content">
              <div className="hero-kicker">
                <Radar className="w-4 h-4" />
                大创项目答辩演示系统
              </div>
              <h1 className="hero-title">
                从流量特征到
                <span>攻击识别与链路解释</span>
              </h1>
              <p className="hero-copy">
                将模型推理、批量检测、事件审计、指标展示和 MITRE ATT&CK 知识图谱整合到一个可演示的安全分析工作台。
              </p>
              <div className="hero-actions">
                <button onClick={() => loadDemoSample("normal")} className="action-button action-safe">
                  加载正常样例
                </button>
                <button onClick={() => loadDemoSample("attack")} className="action-button action-danger">
                  加载攻击样例
                </button>
                <button onClick={generateRandomFeatures} className="action-button action-ghost">
                  随机特征
                </button>
              </div>
              <div className="demo-flow">
                <span>样本输入</span>
                <span>模型检测</span>
                <span>攻击解释</span>
                <span>结果导出</span>
              </div>
            </div>
          </div>

          <div className="command-card">
            <div className="section-eyebrow">System Snapshot</div>
            <div className="command-head">
              <h2>系统概览</h2>
              <span>{apiStatus === "online" ? "运行正常" : "等待后端"}</span>
            </div>
            <div className="overview-grid">
              <InfoCard icon={Brain} label="模型结构" value={metadata?.architecture ?? "未加载"} />
              <InfoCard icon={Database} label="输入维度" value={`${expectedFeatureCount} 维`} />
              <InfoCard icon={BarChart3} label="测试 F1" value={formatPercent(metrics?.test_results?.f1_score)} />
              <InfoCard icon={Network} label="知识图谱" value={`${knowledgeGraph?.entity_count ?? 0} / ${knowledgeGraph?.relation_count ?? 0}`} />
              <InfoCard icon={FileText} label="预处理工件" value={metadata?.preprocessor_loaded ? "已加载" : "未加载"} />
              <InfoCard icon={AlertTriangle} label="检测事件" value={`${eventSummary?.total_events ?? 0} 条`} />
              <InfoCard icon={Database} label="数据样本" value={`${datasetSummary?.rows ?? 0} 条`} />
              <InfoCard icon={CheckCircle} label="数据质量" value={`${datasetProfile?.quality_score ?? 0} 分`} />
            </div>
          </div>
        </section>

        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-2xl text-red-200 text-sm">
            {error}
          </div>
        )}

        <section className="grid lg:grid-cols-[1fr_0.9fr] gap-6 mb-6">
          <Panel title="单条流量检测" icon={Activity}>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              网络流量特征 ({expectedFeatureCount}维，用逗号或空格分隔)
            </label>
            <textarea
              value={features}
              onChange={(event) => setFeatures(event.target.value)}
              placeholder="例如: 0.1, 0.2, 0.3, ..."
              className="w-full h-36 bg-black/30 border border-white/10 rounded-2xl px-4 py-3 text-slate-100 placeholder-slate-500 focus:border-cyan-400 focus:outline-none resize-none font-mono text-sm"
            />
            <button
              onClick={handlePredict}
              disabled={loading || apiStatus === "offline"}
              className="mt-4 w-full px-6 py-3 bg-cyan-300 hover:bg-cyan-200 disabled:bg-slate-700 disabled:text-slate-400 text-slate-950 font-bold rounded-2xl transition-colors flex items-center justify-center gap-2"
            >
              <Shield className="w-4 h-4" />
              {loading ? "检测中..." : "开始检测"}
            </button>
          </Panel>

          <Panel title="检测结果与解释" icon={AlertTriangle}>
            {result ? <PredictionCard result={result} /> : <EmptyState text="请先在左侧输入特征并执行检测。" />}
          </Panel>
        </section>

        <section className="grid lg:grid-cols-[1fr_0.9fr] gap-6 mb-6">
          <Panel title="CSV 批量检测" icon={Upload}>
            <div className="flex gap-3 mb-3">
              <button onClick={generateDemoCsv} className="px-4 py-2 rounded-xl border border-white/15 hover:bg-white/10 text-sm">
                生成示例 CSV
              </button>
              <button
                onClick={handleCsvPredict}
                disabled={csvLoading || (!csvText.trim() && !csvFile) || apiStatus === "offline"}
                className="px-4 py-2 rounded-xl bg-emerald-300 text-slate-950 font-semibold disabled:bg-slate-700 disabled:text-slate-400 text-sm"
              >
                {csvLoading ? "检测中..." : "批量检测"}
              </button>
              <button
                onClick={exportCsvResult}
                disabled={!csvText.trim() && !csvFile}
                className="px-4 py-2 rounded-xl border border-white/15 text-slate-200 hover:bg-white/10 disabled:text-slate-600 disabled:hover:bg-transparent text-sm flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                导出结果
              </button>
            </div>
            <label className="mb-3 flex items-center justify-between gap-3 rounded-2xl border border-dashed border-white/15 bg-black/20 px-4 py-3 text-sm cursor-pointer hover:bg-white/5">
              <span className="text-slate-300 truncate">
                {csvFile ? `已选择文件：${csvFile.name}` : "选择本地 CSV 文件，或直接在下方粘贴 CSV 文本"}
              </span>
              <span className="px-3 py-1 rounded-xl bg-white/10 text-cyan-100">浏览</span>
              <input
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(event) => setCsvFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <textarea
              value={csvText}
              onChange={(event) => {
                setCsvText(event.target.value);
                if (event.target.value.trim()) setCsvFile(null);
              }}
              placeholder={`粘贴 CSV，至少包含 ${expectedFeatureCount} 个数值特征列`}
              className="w-full h-48 bg-black/30 border border-white/10 rounded-2xl px-4 py-3 text-slate-100 placeholder-slate-500 focus:border-emerald-400 focus:outline-none resize-none font-mono text-xs"
            />
          </Panel>

          <Panel title="批量检测汇总" icon={FileText}>
            {csvResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <StatPill label="总行数" value={csvResult.total_rows} />
                  <StatPill label="已处理" value={csvResult.processed_rows} />
                  <StatPill label="攻击数" value={csvResult.summary.Attack ?? 0} />
                </div>
                <div className="space-y-2 max-h-56 overflow-auto pr-1">
                  {csvResult.results.slice(0, 10).map((item, index) => (
                    <div key={index} className="flex items-center justify-between rounded-xl bg-black/25 border border-white/10 px-3 py-2 text-sm">
                      <span>第 {index + 1} 行</span>
                      <span className={item.prediction === "Attack" ? "text-red-300" : "text-emerald-300"}>
                        {item.prediction} / {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <EmptyState text="粘贴 CSV 后可查看批量检测统计和前 10 行结果。" />
            )}
          </Panel>
        </section>

        <section className="grid lg:grid-cols-[0.85fr_1.15fr] gap-6 mb-6">
          <Panel title="运行时检测统计" icon={Radar}>
            <div className="grid grid-cols-3 gap-3">
              <StatPill label="事件总数" value={eventSummary?.total_events ?? 0} />
              <StatPill label="攻击事件" value={eventSummary?.prediction_counts?.Attack ?? 0} />
              <StatPill label="高风险" value={eventSummary?.risk_counts?.High ?? 0} />
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={replayDemoTraffic}
                disabled={replayLoading || apiStatus === "offline"}
                className="px-3 py-2 rounded-xl bg-amber-300 text-slate-950 font-semibold disabled:bg-slate-700 disabled:text-slate-400 text-sm flex items-center gap-2"
              >
                <Zap className="w-4 h-4" />
                {replayLoading ? "回放中..." : "回放演示流量"}
              </button>
              <button onClick={exportEvents} className="px-3 py-2 rounded-xl border border-white/15 text-slate-200 hover:bg-white/10 text-sm flex items-center gap-2">
                <Download className="w-4 h-4" />
                导出事件 CSV
              </button>
              <button onClick={clearEvents} className="px-3 py-2 rounded-xl border border-red-400/30 text-red-200 hover:bg-red-500/10 text-sm flex items-center gap-2">
                <Trash2 className="w-4 h-4" />
                清空事件
              </button>
            </div>
            <div className="mt-4 text-sm text-slate-400">
              最新事件：{eventSummary?.latest ? `${eventSummary.latest.prediction} / ${eventSummary.latest.risk_level}` : "暂无"}
            </div>
            {replayResult && (
              <div className="mt-3 rounded-2xl bg-black/25 border border-white/10 p-3 text-sm text-slate-300">
                已回放 {replayResult.processed_rows} 条，攻击 {replayResult.summary.Attack ?? 0} 条，来源 {replayResult.source_file ?? "demo"}。
              </div>
            )}
          </Panel>

          <Panel title="近期告警记录" icon={AlertTriangle}>
            {events.length ? (
              <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {events.map((event) => (
                  <div key={event.id} className="grid grid-cols-[1fr_auto] gap-3 rounded-xl bg-black/25 border border-white/10 px-3 py-2 text-sm">
                    <div>
                      <div className="font-semibold">
                        {event.prediction} / {event.attack_type ?? "unknown"} / {event.risk_level ?? "Low"}
                      </div>
                      <div className="text-xs text-slate-500">{event.timestamp} · {event.source}</div>
                    </div>
                    <div className={event.prediction === "Attack" ? "text-red-300" : "text-emerald-300"}>
                      {(event.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState text="暂无检测事件。执行单条或批量检测后会自动记录。" />
            )}
          </Panel>
        </section>

        <section className="grid lg:grid-cols-2 gap-6">
          <Panel title="模型评估指标" icon={BarChart3}>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(metrics?.test_results ?? {}).slice(0, 6).map(([key, value]) => (
                <StatPill key={key} label={metricLabelMap[key] ?? key} value={formatPercent(value)} />
              ))}
            </div>
            <div className="mt-4 text-sm text-slate-400">
              最新验证准确率：{formatModelPercent(latestValAcc ?? metadata?.best_val_acc)}
            </div>
          </Panel>

          <Panel title="数据集与知识图谱" icon={Network}>
            <div className="mb-5 rounded-2xl bg-black/25 border border-white/10 p-4">
              <div className="grid grid-cols-3 gap-3 mb-3">
                <StatPill label="样本数" value={datasetSummary?.rows ?? 0} />
                <StatPill label="特征列" value={datasetSummary?.feature_columns ?? 0} />
                <StatPill label="类别数" value={Object.keys(datasetSummary?.label_distribution ?? {}).length} />
              </div>
              <div className="grid grid-cols-3 gap-3 mb-3">
                <StatPill label="质量分" value={datasetProfile?.quality_score ?? 0} />
                <StatPill label="缺失率" value={`${((datasetProfile?.missing_rate ?? 0) * 100).toFixed(2)}%`} />
                <StatPill label="类别失衡" value={(datasetProfile?.imbalance_ratio ?? 0).toFixed(2)} />
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(datasetSummary?.label_distribution ?? {}).map(([label, count]) => (
                  <span key={label} className="px-3 py-1 rounded-full bg-emerald-300/10 border border-emerald-200/20 text-emerald-100 text-sm">
                    {label}: {count}
                  </span>
                ))}
              </div>
              {datasetProfile?.warnings?.length ? (
                <div className="mt-3 space-y-1">
                  {datasetProfile.warnings.map((warning) => (
                    <div key={warning} className="text-xs text-amber-200">
                      {warning}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-3 text-xs text-emerald-200">数据质量审计未发现阻塞性问题。</div>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {(knowledgeGraph?.tactics ?? []).slice(0, 14).map((tactic) => (
                <span key={tactic.id} className="px-3 py-1 rounded-full bg-cyan-300/10 border border-cyan-200/20 text-cyan-100 text-sm">
                  {tactic.name_cn}
                </span>
              ))}
            </div>
          </Panel>
        </section>

        <section className="grid lg:grid-cols-[0.9fr_1.1fr] gap-6 mt-6">
          <Panel title="系统运维状态" icon={Shield}>
            <div className="grid grid-cols-3 gap-3">
              <StatPill label="模型" value={runtimeStatus?.model_loaded ? "在线" : "异常"} />
              <StatPill label="预处理器" value={runtimeStatus?.preprocessor_loaded ? "在线" : "异常"} />
              <StatPill label="知识图谱" value={runtimeStatus?.knowledge_graph_loaded ? "在线" : "异常"} />
            </div>
            <button
              onClick={reloadRuntime}
              disabled={reloadLoading || apiStatus === "offline"}
              className="mt-4 w-full px-4 py-2 rounded-xl bg-cyan-300 text-slate-950 font-semibold disabled:bg-slate-700 disabled:text-slate-400"
            >
              {reloadLoading ? "重载中..." : "重载模型与工件"}
            </button>
            <div className="mt-3 text-xs text-slate-500">
              替换模型、预处理器或知识图谱后可重载运行时，无需重启服务。
            </div>
          </Panel>

          <Panel title="交付物下载" icon={Download}>
            <div className="grid sm:grid-cols-2 gap-3">
              <ArtifactButton label="项目报告" path="/artifacts/report" onClick={openArtifact} />
              <ArtifactButton label="答辩手册" path="/artifacts/runbook" onClick={openArtifact} />
              <ArtifactButton label="模型清单" path="/artifacts/manifest.json" onClick={openArtifact} />
              <ArtifactButton label="数据画像" path="/artifacts/data-profile.json" onClick={openArtifact} />
              <ArtifactButton label="接口规范" path="/artifacts/openapi.json" onClick={openArtifact} />
              <ArtifactButton label="验收清单" path="/artifacts/acceptance-checklist" onClick={openArtifact} />
              <ArtifactButton label="完成度审计" path="/artifacts/completion-audit" onClick={openArtifact} />
              <ArtifactButton label="发布包" path="/artifacts/release.zip" onClick={openArtifact} />
              <ArtifactButton label="发布清单" path="/artifacts/release-manifest.json" onClick={openArtifact} />
            </div>
            <div className="mt-4 grid sm:grid-cols-2 gap-2">
              {Object.entries(runtimeStatus?.artifacts ?? {}).map(([name, artifact]) => (
                <div key={name} className="flex items-center justify-between rounded-xl bg-black/25 border border-white/10 px-3 py-2 text-xs">
                  <span className="text-slate-300">{artifact.path}</span>
                  <span className={artifact.exists ? "text-emerald-300" : "text-red-300"}>
                    {artifact.exists ? `${Math.max(artifact.size, 1)} B` : "缺失"}
                  </span>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      </main>
    </div>
  );
}

function InfoCard({ icon: Icon, label, value }: { icon: typeof Shield; label: string; value: string }) {
  return (
    <div className="info-card">
      <div className="info-icon">
        <Icon className="w-4 h-4" />
      </div>
      <div className="info-label">{label}</div>
      <div className="info-value">{value}</div>
    </div>
  );
}

function Panel({ title, icon: Icon, children }: { title: string; icon: typeof Shield; children: React.ReactNode }) {
  return (
    <div className="panel-card">
      <div className="panel-title-row">
        <div className="panel-icon">
          <Icon className="w-5 h-5" />
        </div>
        <h2>{title}</h2>
      </div>
      {children}
    </div>
  );
}

function PredictionCard({ result }: { result: PredictionResult }) {
  const isAttack = result.prediction === "Attack";
  const chain = result.attack_chain_cn ?? [];
  const classProbabilities = Object.entries(result.class_probabilities ?? {}).sort((a, b) => b[1] - a[1]);
  const featureContributions = result.feature_contributions ?? [];

  return (
    <div className="space-y-4">
      <div className={`rounded-2xl p-4 border ${isAttack ? "bg-red-500/10 border-red-400/30" : "bg-emerald-500/10 border-emerald-400/30"}`}>
        <div className="flex items-center gap-2 text-lg font-bold">
          {isAttack ? <AlertTriangle className="w-6 h-6 text-red-300" /> : <CheckCircle className="w-6 h-6 text-emerald-300" />}
          {isAttack ? "检测到攻击行为" : "正常流量"}
        </div>
        <div className="mt-2 text-sm text-slate-300">
          风险等级：{result.risk_level ?? "Low"} / 置信度：{(result.confidence * 100).toFixed(2)}%
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatPill label="正常概率" value={`${(result.probabilities.Normal * 100).toFixed(2)}%`} />
        <StatPill label="攻击概率" value={`${(result.probabilities.Attack * 100).toFixed(2)}%`} />
      </div>

      <div className="rounded-2xl bg-black/25 border border-white/10 p-4">
        <div className="text-sm text-slate-400 mb-2">处置建议</div>
        <p className="text-slate-200 text-sm leading-6">{result.recommendation || "暂无建议"}</p>
      </div>

      <div className="rounded-2xl bg-black/25 border border-white/10 p-4">
        <div className="text-sm text-slate-400 mb-3">攻击链还原</div>
        {chain.length ? (
          <div className="flex flex-wrap items-center gap-2">
            {chain.map((step, index) => (
              <div key={step} className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-full bg-amber-300/10 border border-amber-200/20 text-amber-100 text-sm">{step}</span>
                {index < chain.length - 1 && <Zap className="w-4 h-4 text-amber-300" />}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-slate-500">当前结果未映射到攻击链。</div>
        )}
      </div>

      <div className="rounded-2xl bg-black/25 border border-white/10 p-4">
        <div className="text-sm text-slate-400 mb-3">类别概率 Top</div>
        <div className="space-y-2">
          {classProbabilities.slice(0, 4).map(([label, value]) => (
            <div key={label}>
              <div className="flex justify-between text-xs mb-1">
                <span>{label}</span>
                <span>{(value * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div className="h-full bg-cyan-300" style={{ width: `${value * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-2xl bg-black/25 border border-white/10 p-4">
        <div className="text-sm text-slate-400 mb-3">关键特征贡献</div>
        <div className="space-y-2">
          {featureContributions.slice(0, 5).map((item) => (
            <div key={`${item.feature}-${item.index}`}>
              <div className="flex justify-between text-xs mb-1">
                <span>{item.feature}</span>
                <span>{item.value.toFixed(3)}</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div className="h-full bg-amber-300" style={{ width: `${Math.min(item.score * 20, 100)}%` }} />
              </div>
            </div>
          ))}
          {!featureContributions.length && <div className="text-sm text-slate-500">暂无特征贡献数据。</div>}
        </div>
      </div>
    </div>
  );
}

function StatPill({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-pill">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      {text}
    </div>
  );
}

function ArtifactButton({ label, path, onClick }: { label: string; path: string; onClick: (path: string) => void }) {
  return (
    <button
      onClick={() => onClick(path)}
      className="artifact-button"
    >
      <div className="artifact-label">{label}</div>
      <div className="artifact-path">{path}</div>
    </button>
  );
}

export default App;
