import { useEffect } from "react";

import { useCreateSession, useDeleteSession, useSessionMessages, useSessions } from "../../api/sessions";
import { useSessionStore } from "../../store/sessionStore";
import { NewChatButton } from "../sidebar/NewChatButton";
import { SessionList } from "../sidebar/SessionList";

export function Sidebar() {
  const sessions = useSessionStore((state) => state.sessions);
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const setSessions = useSessionStore((state) => state.setSessions);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const setMessages = useSessionStore((state) => state.setMessages);
  const removeSession = useSessionStore((state) => state.removeSession);
  const sessionsQuery = useSessions();
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();
  const messagesQuery = useSessionMessages(activeSessionId);

  useEffect(() => {
    if (sessionsQuery.data) {
      setSessions(sessionsQuery.data);
    }
  }, [sessionsQuery.data, setSessions]);

  useEffect(() => {
    if (messagesQuery.data) {
      setMessages(messagesQuery.data);
    }
  }, [messagesQuery.data, setMessages]);

  const newConversation = (): void => {
    createSession.mutate(undefined, {
      onSuccess: (session) => {
        setActiveSession(session.id);
        setMessages([]);
      },
    });
  };

  const deleteConversation = (sessionId: string): void => {
    deleteSession.mutate(sessionId, {
      onSuccess: () => removeSession(sessionId),
    });
  };

  return (
    <aside
      className="hidden md:flex"
      style={{
        width: "var(--sidebar-width)",
        minWidth: "var(--sidebar-width)",
        flexDirection: "column",
        height: "100%",
        background: "var(--bg-surface)",
        borderRight: "1px solid var(--border-subtle)",
        padding: "0",
      }}
    >
      {/* Logo / App Header */}
      <div
        style={{
          padding: "16px 14px 12px",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: 14,
            color: "#fff",
            flexShrink: 0,
            letterSpacing: "-0.5px",
          }}
        >
          Q
        </div>
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--text-primary)",
              letterSpacing: "0.01em",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            Quadtrix.cpp
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>char-level GPT v1.0</div>
        </div>
      </div>

      {/* New Chat */}
      <div style={{ padding: "10px 10px 6px" }}>
        <NewChatButton onClick={newConversation} />
      </div>

      {/* Recent label */}
      <div
        style={{
          padding: "8px 14px 4px",
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        Recent
      </div>

      {/* Session list */}
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "2px 6px" }}>
        <SessionList
          activeSessionId={activeSessionId}
          onDelete={deleteConversation}
          onSelect={setActiveSession}
          sessions={sessions}
        />
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "10px 14px",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: 11,
          color: "var(--text-muted)",
        }}
      >
        API Configuration
      </div>
    </aside>
  );
}
