import { Card } from "@/src/components/ui/Card";

export default function AboutPage() {
  return (
    <div className="space-y-5">
      <Card title="Young Counsel" subtitle="Legal Intelligence Systems">
        <p className="text-sm leading-relaxed text-text-muted">
          Young Counsel is a local-first legal intelligence workspace designed to
          help practitioners rapidly review, explore, and compare complex
          contracts and case files. It combines retrieval-augmented generation,
          clause-level analytics, and deterministic document IDs so you can trace
          every insight back to specific language in the source documents.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-text-muted">
          The system runs entirely in your environment, using the existing Legal
          Document Intelligence pipeline in this repository. Documents are
          ingested, parsed, chunked, and stored locally for repeatable analysis
          and auditability.
        </p>
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        <Card title="Contract Review">
          Quickly surface an executive summary, key risk areas, and supporting
          evidence blocks for complex agreements.
        </Card>
        <Card title="Document Explorer">
          Navigate clause-level evidence, filter by risk or topic, and ground
          LLM answers in specific contract language.
        </Card>
        <Card title="Due Diligence Support">
          Use structured clause extraction and comparison to build consistent,
          evidence-backed due diligence notes.
        </Card>
      </div>

      <Card title="Disclaimer">
        <p className="text-sm leading-relaxed text-text-muted">
          Young Counsel is an analytical and productivity tool. It does not
          provide legal advice, does not replace qualified counsel, and should
          not be relied upon as the sole basis for legal decisions. All insights
          and summaries must be reviewed and validated by a licensed legal
          professional.
        </p>
      </Card>
    </div>
  );
}

