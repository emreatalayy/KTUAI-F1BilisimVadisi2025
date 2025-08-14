import { ReactNode } from 'react'

export function Tablo({ basliklar, satirlar }: { basliklar: string[]; satirlar: ReactNode[][] }) {
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-metinIkincil">
            {basliklar.map((b) => (
              <th key={b} className="text-left font-medium py-2 px-3 border-b border-cizgi bg-arka">
                {b}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {satirlar.map((hucreler, i) => (
            <tr key={i} className="h-11 border-b border-cizgi">
              {hucreler.map((h, j) => (
                <td key={j} className="px-3">
                  {h}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


