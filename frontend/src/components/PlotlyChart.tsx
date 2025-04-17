import React from 'react';
import Plot from 'react-plotly.js';

interface PlotlyChartProps {
  visualizationJson?: string;
  className?: string;
}

export function PlotlyChart({ visualizationJson, className = '' }: PlotlyChartProps) {
  if (!visualizationJson) {
    return null;
  }

  try {
    const plotData = JSON.parse(visualizationJson);
    
    return (
      <div className={`w-full ${className}`}>
        <Plot
          data={plotData.data}
          layout={{
            ...plotData.layout,
            autosize: true,
            margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
              color: document.documentElement.classList.contains('dark') 
                ? '#E5E4DF' // neutral-300
                : '#40403E'  // neutral-700
            }
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: [
              'sendDataToCloud',
              'autoScale2d',
              'hoverClosestCartesian',
              'hoverCompareCartesian',
              'lasso2d',
              'select2d'
            ]
          }}
          style={{ width: '100%', height: '400px' }}
          className="w-full"
        />
      </div>
    );
  } catch (error) {
    console.error('Error parsing visualization JSON:', error);
    return (
      <div className={`text-center p-4 text-red-500 ${className}`}>
        Error rendering visualization
      </div>
    );
  }
} 