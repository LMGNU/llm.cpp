import { motion } from "framer-motion";
import { useSettingsStore } from "../../store/settingsStore";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Slider } from "../ui/Slider";

export function SettingsPanel() {
  const {
    apiBaseUrl, maxTokens, temperature, modelLabel, modelBackend, settingsOpen,
    setApiBaseUrl, setMaxTokens, setTemperature, setModelLabel, setModelBackend, setSettingsOpen,
  } = useSettingsStore();

  if (!settingsOpen) return null;

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
        padding: "0",
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
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>Settings</span>
        <Button onClick={() => setSettingsOpen(false)} style={{ height: 28, padding: "0 8px", fontSize: 11 }}>
          Close
        </Button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Input label="Backend URL" onChange={(event) => setApiBaseUrl(event.target.value)} value={apiBaseUrl} />
          <label style={{ display: "block", fontSize: 12, color: "var(--text-secondary)" }}>
            <span style={{ display: "block", marginBottom: 6, fontWeight: 500 }}>Active model</span>
            <select
              style={{
                width: "100%",
                height: 36,
                borderRadius: 6,
                border: "1px solid var(--border-muted)",
                background: "var(--bg-elevated)",
                color: "var(--text-primary)",
                padding: "0 10px",
                fontSize: 12,
                outline: "none",
              }}
              onChange={(event) => setModelBackend(event.target.value === "torch" ? "torch" : "cpp")}
              value={modelBackend}
            >
              <option value="cpp">C++ server</option>
              <option value="torch">PyTorch .pt checkpoint</option>
            </select>
          </label>
          <Slider
            label="Max new tokens"
            max={500}
            min={20}
            onChange={(event) => setMaxTokens(Number(event.target.value))}
            value={maxTokens}
            valueLabel={String(maxTokens)}
          />
          <Slider
            label="Temperature"
            max={2}
            min={0.1}
            onChange={(event) => setTemperature(Number(event.target.value))}
            step={0.1}
            value={temperature}
            valueLabel={temperature.toFixed(1)}
          />
          <Input label="Model label" onChange={(event) => setModelLabel(event.target.value)} value={modelLabel} />
        </div>
      </div>
    </motion.aside>
  );
}
