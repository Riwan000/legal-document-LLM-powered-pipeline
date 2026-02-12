import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Young Counsel",
  description: "Legal intelligence system UI for Young Counsel"
};

export default function RootLayout(props: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-text">
        {props.children}
      </body>
    </html>
  );
}

