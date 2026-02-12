"use client";

import { useState, useRef, useEffect, type ChangeEvent } from "react";
import { Card } from "@/src/components/ui/Card";
import { uploadDocument, listDocuments, deleteDocument, type DocumentSummary } from "@/src/lib/api";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<"document" | "template">("document");
  const [displayName, setDisplayName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<Record<string, unknown> | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const loadDocuments = () => {
    listDocuments()
      .then((list) => setDocuments(Array.isArray(list) ? list : []))
      .catch(() => setDocuments([]));
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadResult(null);
      setUploadError(null);
      if (!displayName.trim()) setDisplayName(file.name.replace(/\.[^/.]+$/, ""));
    }
  };

  const handleSelectClick = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploadError(null);
    setUploadResult(null);
    setUploading(true);
    try {
      const result = await uploadDocument(selectedFile, {
        document_type: documentType,
        display_name: displayName.trim() || undefined,
      });
      setUploadResult(result);
      loadDocuments();
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    if (!documentId) return;
    if (!window.confirm("Are you sure you want to permanently delete this document? This action cannot be undone.")) {
      return;
    }
    setDeleteError(null);
    setDeletingId(documentId);
    try {
      await deleteDocument(documentId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== documentId));
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : "Delete failed.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-sm font-semibold tracking-wide text-text">
          DOCUMENT UPLOAD
        </h2>
        <p className="mt-1 text-xs text-text-muted">
          Upload PDF or DOCX to parse, chunk, embed, and index. Documents are
          processed locally by Young Counsel.
        </p>
      </section>

      <div className="rounded-xl border border-dashed border-border bg-background-elevated-soft/40 p-8 text-center">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx"
          className="hidden"
          onChange={handleFileChange}
        />
        <p className="text-sm font-medium text-text">
          {selectedFile
            ? selectedFile.name
            : "Drag and drop files here, or click to select"}
        </p>
        <p className="mt-2 text-xs text-text-muted">
          Supported formats: PDF, DOCX, DOC
        </p>
        <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            className="inline-flex h-9 items-center rounded-lg bg-accent px-4 text-xs font-semibold text-black shadow-card hover:bg-accent-soft"
            onClick={handleSelectClick}
          >
            Select File
          </button>
          <button
            type="button"
            disabled={!selectedFile || uploading}
            onClick={handleUpload}
            className="inline-flex h-9 items-center rounded-lg border border-border bg-background-elevated px-4 text-xs font-semibold text-text-muted disabled:opacity-60 hover:bg-background-elevated-soft"
          >
            {uploading ? "Uploading…" : "Upload & Ingest"}
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <label className="text-xs text-text-muted">Document type</label>
          <div className="mt-1 flex gap-4">
            <label className="flex items-center gap-2 text-sm text-text">
              <input
                type="radio"
                name="document_type"
                checked={documentType === "document"}
                onChange={() => setDocumentType("document")}
                className="border-border bg-background-elevated text-accent focus:ring-accent"
              />
              Document
            </label>
            <label className="flex items-center gap-2 text-sm text-text">
              <input
                type="radio"
                name="document_type"
                checked={documentType === "template"}
                onChange={() => setDocumentType("template")}
                className="border-border bg-background-elevated text-accent focus:ring-accent"
              />
              Template
            </label>
          </div>
        </div>
        <div>
          <label className="text-xs text-text-muted">Display name (optional)</label>
          <input
            type="text"
            className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text placeholder:text-text-subtle outline-none focus:border-accent"
            placeholder="e.g. Employment Contract (2023)"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
        </div>
      </div>

      {uploadError && (
        <p className="text-sm text-risk-high">{uploadError}</p>
      )}

      {uploadResult && (
        <Card title="FILE METADATA CONFIRMATION">
          <dl className="grid gap-3 text-xs text-text-muted sm:grid-cols-2">
            {Object.entries(uploadResult).map(([key, value]) => (
              <div key={key}>
                <dt className="text-text-subtle">{key}</dt>
                <dd className="mt-0.5 text-text">
                  {typeof value === "object" && value !== null
                    ? JSON.stringify(value).slice(0, 120)
                    : String(value ?? "").slice(0, 120)}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-xs text-text-subtle">
            Documents are processed locally and never leave your environment.
          </p>
        </Card>
      )}

      <Card title="Uploaded documents">
        {documents.length === 0 ? (
          <p className="text-sm text-text-muted">No documents uploaded yet.</p>
        ) : (
          <ul className="space-y-2 text-sm text-text-muted">
            {documents.map((doc) => (
              <li
                key={doc.document_id}
                className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background-elevated-soft/50 px-3 py-2"
              >
                <div>
                  <span className="font-medium text-text">{doc.filename}</span>
                  <span className="ml-2 text-xs text-text-subtle">
                    ({doc.document_id})
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(doc.document_id)}
                  disabled={deletingId === doc.document_id}
                  className="inline-flex h-8 items-center rounded-lg border border-risk-high/40 bg-background-elevated px-3 text-xs font-semibold text-risk-high hover:bg-risk-high/5 disabled:opacity-60"
                >
                  {deletingId === doc.document_id ? "Deleting…" : "Delete"}
                </button>
              </li>
            ))}
          </ul>
        )}
        {deleteError && (
          <p className="mt-3 text-sm text-risk-high">{deleteError}</p>
        )}
      </Card>
    </div>
  );
}
