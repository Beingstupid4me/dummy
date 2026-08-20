import { AnalysisView } from "@/components/views/analysis";

export const metadata = { title: "Analysis" };

export default async function AnalysisCasePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <AnalysisView id={id} />;
}
