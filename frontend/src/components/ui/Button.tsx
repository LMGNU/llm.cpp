import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "danger";
  children: ReactNode;
}

export function Button({ variant = "ghost", className = "", children, style, ...props }: ButtonProps) {
  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    height: 32,
    padding: "0 10px",
    borderRadius: 6,
    fontSize: 12,
    fontWeight: 500,
    cursor: "pointer",
    transition: "background 0.12s, border-color 0.12s, color 0.12s",
  };

  const variants: Record<string, React.CSSProperties> = {
    primary: {
      background: "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
      border: "none",
      color: "#fff",
    },
    ghost: {
      background: "transparent",
      border: "1px solid var(--border-muted)",
      color: "var(--text-secondary)",
    },
    danger: {
      background: "transparent",
      border: "1px solid rgba(224,82,82,0.3)",
      color: "var(--red)",
    },
  };

  return (
    <button
      style={{ ...base, ...variants[variant], ...style }}
      onMouseEnter={(e) => {
        const el = e.currentTarget;
        if (variant === "ghost") {
          el.style.background = "var(--bg-elevated)";
          el.style.borderColor = "var(--border-strong)";
          el.style.color = "var(--text-primary)";
        } else if (variant === "danger") {
          el.style.background = "rgba(224,82,82,0.1)";
        }
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget;
        if (variant === "ghost") {
          el.style.background = "transparent";
          el.style.borderColor = "var(--border-muted)";
          el.style.color = "var(--text-secondary)";
        } else if (variant === "danger") {
          el.style.background = "transparent";
        }
      }}
      {...props}
    >
      {children}
    </button>
  );
}
