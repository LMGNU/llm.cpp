import { motion } from "framer-motion";
import { useStats } from "../../api/health";
import { useSettingsStore } from "../../store/settingsStore";
import { formatDuration } from "../../utils/time";
import { Button } from "../ui/Button";

export function StatsPanel() {
  const statsOpen = useSettingsStore((state) => state.statsOpen);
  const setStatsOpen = useSettingsStore((state) => state.setStatsOpen);
  const { data, isLoading, isError } = useStats();

  if (!statsOpen) return null;

  const rows = data
    ? [
        ["Model", data.model],
        ["Architecture", data.architecture],
        ["Parameters", data.parameters.toLocaleString()],
        ["Vocabulary", `${data.vocabulary} characters`],
        ["Val Loss", `${data.val_loss} nats`],
        ["Context", `${data.context} characters`],
        ["Training", data.training],
        ["C++ Backend", `${data.backend} ${data.backend_online ? "ok" : "offline"}`],
        ["PyTorch", `${data.torch_checkpoint} ${data.torch_online ? "ok" : "missing"}`],
        ["Uptime", formatDuration(data.uptime_seconds)],
      ]
    : [];

  return (
    <motion.aside
      animate={{ x: 0 }}
      initial={{ x: 280 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      style={{
        position: "fixed",
        right: 0,
        top: 0,
        zIndex: 40,
        height: "100%",
        width: 280,
        borderLeft: "1px solid var(--border-subtle)",
        background: "var(--bg-surface)",
        boxShadow: "-16px 0 48px rgba(0,0,0,0.4)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 16px",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>Model Stats</span>
        <Button onClick={() => setStatsOpen(false)} style={{ height: 28, padding: "0 8px", fontSize: 11 }}>
          Close
        </Button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
        {isLoading && (
          <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "8px 0" }}>Loading stats…</div>
        )}
        {isError && (
          <div style={{ fontSize: 12, color: "var(--red)", padding: "8px 0" }}>Stats unavailable</div>
        )}
        {rows.map(([label, value]) => (
          <div
            key={label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              gap: 12,
              padding: "9px 0",
              borderBottom: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ fontSize: 11, color: "var(--text-muted)", flexShrink: 0 }}>{label}</span>
            <span style={{ fontSize: 11, color: "var(--text-primary)", textAlign: "right", wordBreak: "break-all" }}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </motion.aside>
  );
}
