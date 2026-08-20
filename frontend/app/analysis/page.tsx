"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { AnalysisView } from "@/components/views/analysis";
import { useConsole } from "@/lib/store";

export default function AnalysisIndexPage() {
  const { selected } = useConsole();
  const router = useRouter();

  useEffect(() => {
    router.replace(`/analysis/${selected.id}`);
  }, [router, selected.id]);

  return <AnalysisView />;
}
