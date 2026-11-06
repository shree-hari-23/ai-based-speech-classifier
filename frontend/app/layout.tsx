import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import { AuthProvider } from "@/lib/auth-context";
import "./globals.css";

export const metadata: Metadata = {
  title: "EmotiSound AI – Acoustic Speech Emotion Recognition",
  description: "Machine Learning & Deep Learning Speech Emotion Recognition System (CNN, SVM, MLP)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased flex flex-col font-sans">
        <AuthProvider>
          <Navbar />
          <div className="flex-1 flex flex-row">
            <Sidebar />
            <main className="flex-1 p-6 lg:p-8 max-w-7xl mx-auto w-full">
              {children}
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}

