import './globals.css'

export const metadata = {
  title: 'TIMEKEEPER | Historical Photo Restoration',
  description: 'Confidence-Calibrated Restoration and Colorization of Damaged Historical Photographs',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  )
}
