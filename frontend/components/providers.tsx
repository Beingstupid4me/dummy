"use client";

import { ConsoleProvider } from "@/lib/store";
import { Shell } from "@/components/shell";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ConsoleProvider>
      <Shell>{children}</Shell>
    </ConsoleProvider>
  );
}
