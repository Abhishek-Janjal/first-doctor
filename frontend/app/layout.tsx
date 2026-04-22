import "./globals.css";
import Providers from "./providers";

export const metadata = {
  title: "AI Health Assistant",
  description: "Early diagnosis assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}