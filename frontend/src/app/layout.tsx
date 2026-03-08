import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["300", "400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "NanoPrune Canvas — Neural Network Pruning & Quantization Visualizer",
  description:
    "Visually simulate magnitude-based pruning and post-training quantization on neural networks to estimate edge-device compatibility.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${spaceGrotesk.variable} ${jetbrains.variable} font-sans antialiased bg-[var(--bg-root)] text-[var(--text-primary)]`}>
        {children}
      </body>
    </html>
  );
}

