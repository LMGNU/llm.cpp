import { Button } from "../ui/Button";

interface NewChatButtonProps {
  onClick: () => void;
}

export function NewChatButton({ onClick }: NewChatButtonProps) {
  return (
    <button
      onClick={onClick}
      type="button"
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "7px 10px",
        borderRadius: 7,
        border: "1px solid var(--border-muted)",
        background: "var(--bg-elevated)",
        color: "var(--text-primary)",
        fontSize: 13,
        fontWeight: 500,
        cursor: "pointer",
        transition: "background 0.15s, border-color 0.15s",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = "var(--bg-hover)";
        (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border-strong)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = "var(--bg-elevated)";
        (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border-muted)";
      }}
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
      <span>New conversation</span>
    </button>
  );
}
