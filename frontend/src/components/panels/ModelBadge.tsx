import { Tooltip } from "../ui/Tooltip";

interface ModelBadgeProps {
  online: boolean;
  degraded: boolean;
  label: string;
  tooltip: string;
}

export function ModelBadge({ online, degraded, label, tooltip }: ModelBadgeProps) {
  const dotColor = online
    ? "var(--status-online)"
    : degraded
    ? "var(--yellow)"
    : "var(--red)";

  return (
    <Tooltip label={tooltip}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          height: 32,
          padding: "0 10px",
          borderRadius: 6,
          border: "1px solid var(--border-muted)",
          background: "var(--bg-elevated)",
          fontSize: 12,
          color: "var(--text-secondary)",
          cursor: "default",
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: dotColor,
            boxShadow: online ? `0 0 6px ${dotColor}` : "none",
            animation: degraded ? "pulse 1.5s ease-in-out infinite" : "none",
            flexShrink: 0,
          }}
        />
        <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
        <span style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {label}
        </span>
      </div>
    </Tooltip>
  );
}
