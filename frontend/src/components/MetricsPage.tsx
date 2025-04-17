import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { ArrowLeft, BarChart, Clock, RefreshCw, Calendar, CheckCircle, XCircle, Database, LineChart, PieChart } from 'lucide-react';
import { CollapsibleSection } from './CollapsibleSection';
import { Tooltip } from './Tooltip';

interface MetricsPageProps {
  onBack: () => void;
}

export const MetricsPage: React.FC<MetricsPageProps> = ({ onBack }) => {
  const [timeRange, setTimeRange] = useState<'all' | 'day' | 'week' | 'month'>('all');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['metrics', timeRange],
    queryFn: async () => {
      const response = await axios.get(`https://text2sql.fly.dev/metrics?time_range=${timeRange}`);
      return response.data;
    }
  });

  // Utility function to format numbers with commas
  const formatNumber = (num: number) => {
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  // Utility function to format time in ms to a readable format
  const formatTime = (ms: number) => {
    if (ms < 1000) {
      return `${Math.round(ms)} ms`;
    } else if (ms < 60000) {
      return `${(ms / 1000).toFixed(2)} s`;
    } else {
      return `${(ms / 60000).toFixed(2)} min`;
    }
  };

  const renderBarChart = (data: { type: string, count: number }[], title: string, color: string = 'rgba(99, 102, 241, 0.8)') => {
    if (!data || data.length === 0) return null;
    
    // Sort by count descending
    const sortedData = [...data].sort((a, b) => b.count - a.count);
    
    // Get max value for scaling
    const maxCount = Math.max(...sortedData.map(item => item.count));
    
    return (
      <div className="mt-4">
        <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">{title}</h3>
        <div className="space-y-2">
          {sortedData.map((item, index) => (
            <div key={index} className="flex items-center">
              <div className="w-32 text-xs text-neutral-600 dark:text-neutral-400 truncate mr-2">{item.type}</div>
              <div className="flex-1 h-5 bg-neutral-100 dark:bg-neutral-700 rounded-full overflow-hidden">
                <div 
                  className="h-full rounded-full" 
                  style={{ 
                    width: `${(item.count / maxCount) * 100}%`,
                    backgroundColor: color
                  }}
                ></div>
              </div>
              <div className="ml-2 text-xs text-neutral-600 dark:text-neutral-400 min-w-12 text-right">{item.count}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderLineChart = (
    labels: string[], 
    datasets: { label: string, data: number[], color: string }[]
  ) => {
    if (!labels || labels.length === 0) return null;
    
    const height = 200;
    const width = Math.max(labels.length * 40, 300);
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    
    // Find max value for scaling
    const maxValue = Math.max(...datasets.flatMap(dataset => dataset.data)) * 1.1; // 10% padding
    
    // Scale values to fit in the chart
    const scaleY = (value: number) => {
      return height - padding.bottom - (value / maxValue) * (height - padding.top - padding.bottom);
    };
    
    // Generate x positions
    const stepX = (width - padding.left - padding.right) / Math.max(labels.length - 1, 1);
    const xPositions = labels.map((_, i) => padding.left + i * stepX);
    
    return (
      <div className="relative h-[200px] mt-4 overflow-hidden">
        <svg width="100%" height={height} style={{ minWidth: width }} className="overflow-visible">
          {/* Y-axis */}
          <line 
            x1={padding.left} 
            y1={padding.top} 
            x2={padding.left} 
            y2={height - padding.bottom} 
            stroke="currentColor" 
            className="text-neutral-300 dark:text-neutral-700" 
            strokeWidth="1"
          />
          
          {/* X-axis */}
          <line 
            x1={padding.left} 
            y1={height - padding.bottom} 
            x2={width - padding.right} 
            y2={height - padding.bottom} 
            stroke="currentColor"
            className="text-neutral-300 dark:text-neutral-700" 
            strokeWidth="1"
          />
          
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
            const y = scaleY(maxValue * ratio);
            return (
              <React.Fragment key={i}>
                <line 
                  x1={padding.left} 
                  y1={y} 
                  x2={width - padding.right} 
                  y2={y} 
                  stroke="currentColor" 
                  className="text-neutral-200 dark:text-neutral-800" 
                  strokeWidth="1" 
                  strokeDasharray="3,3"
                />
                <text 
                  x={padding.left - 5} 
                  y={y} 
                  textAnchor="end" 
                  dominantBaseline="middle" 
                  className="text-[10px] fill-neutral-500 dark:fill-neutral-400"
                >
                  {formatNumber(maxValue * (1 - ratio))}
                </text>
              </React.Fragment>
            );
          })}
          
          {/* X-axis labels - show every nth label to avoid crowding */}
          {labels.map((label, i) => {
            // Only show every 2nd label if we have a lot of them
            const showLabel = labels.length <= 7 || i % 2 === 0;
            if (!showLabel) return null;
            
            return (
              <text 
                key={i} 
                x={xPositions[i]} 
                y={height - padding.bottom + 15} 
                textAnchor="middle" 
                className="text-[10px] fill-neutral-500 dark:fill-neutral-400 rotate-45"
                transform={`rotate(45, ${xPositions[i]}, ${height - padding.bottom + 15})`}
              >
                {label}
              </text>
            );
          })}
          
          {/* Datasets */}
          {datasets.map((dataset, datasetIndex) => {
            // Create path for the line
            let path = '';
            dataset.data.forEach((value, i) => {
              const x = xPositions[i];
              const y = scaleY(value);
              if (i === 0) {
                path += `M ${x} ${y}`;
              } else {
                path += ` L ${x} ${y}`;
              }
            });
            
            return (
              <React.Fragment key={datasetIndex}>
                {/* Line */}
                <path 
                  d={path} 
                  fill="none" 
                  stroke={dataset.color} 
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                
                {/* Points */}
                {dataset.data.map((value, i) => (
                  <circle 
                    key={i} 
                    cx={xPositions[i]} 
                    cy={scaleY(value)} 
                    r="3" 
                    fill={dataset.color} 
                  />
                ))}
              </React.Fragment>
            );
          })}
        </svg>
        
        {/* Legend */}
        <div className="flex items-center justify-center mt-2 flex-wrap gap-4">
          {datasets.map((dataset, i) => (
            <div key={i} className="flex items-center">
              <div 
                className="w-3 h-3 rounded-full mr-1" 
                style={{ backgroundColor: dataset.color }}
              ></div>
              <span className="text-xs text-neutral-600 dark:text-neutral-400">{dataset.label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderPieChart = (data: { label: string, value: number, color: string }[]) => {
    if (!data || data.length === 0) return null;
    
    const total = data.reduce((acc, item) => acc + item.value, 0);
    const radius = 80;
    const center = { x: radius + 20, y: radius + 20 };
    let startAngle = 0;
    
    // Generate pie slices
    const slices = data.map((item, index) => {
      const percentage = item.value / total;
      const angle = percentage * 360;
      const endAngle = startAngle + angle;
      
      // Convert angles to radians
      const startRad = (startAngle - 90) * Math.PI / 180;
      const endRad = (endAngle - 90) * Math.PI / 180;
      
      // Calculate path
      const x1 = center.x + radius * Math.cos(startRad);
      const y1 = center.y + radius * Math.sin(startRad);
      const x2 = center.x + radius * Math.cos(endRad);
      const y2 = center.y + radius * Math.sin(endRad);
      
      // Create path
      const largeArcFlag = angle > 180 ? 1 : 0;
      const path = `
        M ${center.x} ${center.y}
        L ${x1} ${y1}
        A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}
        Z
      `;
      
      // Calculate label position (middle of arc)
      const labelRad = (startRad + endRad) / 2;
      const labelDistance = radius * 0.7; // slightly inside
      const labelX = center.x + labelDistance * Math.cos(labelRad);
      const labelY = center.y + labelDistance * Math.sin(labelRad);
      
      // Only show label if slice is big enough
      const showLabel = percentage > 0.05;
      
      // Store the current end angle as the next start angle
      startAngle = endAngle;
      
      return {
        path,
        color: item.color,
        label: item.label,
        value: item.value,
        percentage,
        labelX,
        labelY,
        showLabel
      };
    });
    
    return (
      <div className="flex flex-col md:flex-row items-center justify-center gap-4 mt-4">
        <svg width={radius * 2 + 40} height={radius * 2 + 40}>
          {slices.map((slice, i) => (
            <g key={i}>
              <path
                d={slice.path}
                fill={slice.color}
                stroke="white"
                strokeWidth="1"
              />
              {slice.showLabel && (
                <text
                  x={slice.labelX}
                  y={slice.labelY}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs font-bold fill-white"
                >
                  {Math.round(slice.percentage * 100)}%
                </text>
              )}
            </g>
          ))}
        </svg>
        
        <div className="flex flex-col gap-2">
          {slices.map((slice, i) => (
            <div key={i} className="flex items-center">
              <div 
                className="w-3 h-3 rounded-full mr-2" 
                style={{ backgroundColor: slice.color }}
              ></div>
              <span className="text-xs text-neutral-700 dark:text-neutral-300">{slice.label}</span>
              <span className="text-xs text-neutral-500 dark:text-neutral-400 ml-2">
                ({formatNumber(slice.value)} - {Math.round(slice.percentage * 100)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Tooltip content="Return to main view" position="bottom" offset={24}>
              <button 
                onClick={onBack}
                className="flex items-center mr-4 text-neutral-600 dark:text-neutral-300 hover:text-primary"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                <span>Back</span>
              </button>
            </Tooltip>
            <h1 className="text-2xl font-semibold text-neutral-900 dark:text-white">Query Analytics</h1>
          </div>
          
          {/* Time range selector */}
          <div className="flex items-center gap-2">
            <div className="text-sm text-neutral-500 dark:text-neutral-400 mr-2 flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              <span>Time Range:</span>
            </div>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="bg-white dark:bg-neutral-700 border border-neutral-300 dark:border-neutral-600 text-sm rounded-md px-2 py-1"
            >
              <option value="all">All Time</option>
              <option value="day">Last 24 Hours</option>
              <option value="week">Last 7 Days</option>
              <option value="month">Last 30 Days</option>
            </select>
            <Tooltip content="Refresh data" position="bottom" offset={24}>
              <button
                onClick={() => refetch()}
                className="p-1.5 bg-neutral-100 dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded-md hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </Tooltip>
          </div>
        </div>

        {isLoading ? (
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-6">
            <p className="text-neutral-600 dark:text-neutral-300">Loading metrics...</p>
          </div>
        ) : error ? (
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-6 border-l-4 border-red-500">
            <h3 className="text-red-500 font-medium mb-2">Error</h3>
            <p className="text-neutral-700 dark:text-neutral-300">
              Failed to load metrics. Please try again.
            </p>
          </div>
        ) : data?.status === "error" ? (
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-6 border-l-4 border-amber-500">
            <h3 className="text-amber-500 font-medium mb-2">No Data Available</h3>
            <p className="text-neutral-700 dark:text-neutral-300">
              {data.message || "No query history available for the selected time range."}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {/* High-level metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-4">
                <div className="flex items-center mb-2">
                  <Database className="w-5 h-5 text-primary mr-2" />
                  <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Total Queries</h3>
                </div>
                <div className="text-2xl font-semibold text-neutral-900 dark:text-white">
                  {formatNumber(data.total_queries)}
                </div>
              </div>
              
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-4">
                <div className="flex items-center mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                  <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Success Rate</h3>
                </div>
                <div className="text-2xl font-semibold text-neutral-900 dark:text-white">
                  {data.success_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  {data.successful_queries} successful / {data.error_queries} failed
                </div>
              </div>
              
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-4">
                <div className="flex items-center mb-2">
                  <RefreshCw className="w-5 h-5 text-amber-500 mr-2" />
                  <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Retry Rate</h3>
                </div>
                <div className="text-2xl font-semibold text-neutral-900 dark:text-white">
                  {data.retry_metrics.retry_success_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  {data.retry_metrics.successful_after_retry} fixed after retry
                </div>
              </div>
              
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-4">
                <div className="flex items-center mb-2">
                  <Clock className="w-5 h-5 text-blue-500 mr-2" />
                  <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Avg Query Time</h3>
                </div>
                <div className="text-2xl font-semibold text-neutral-900 dark:text-white">
                  {formatTime(data.performance_metrics.avg_total_time_ms)}
                </div>
              </div>
            </div>
            
            {/* Main content sections */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Metrics Section */}
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden">
                <CollapsibleSection 
                  title="Performance Metrics" 
                  icon={<Clock className="w-5 h-5 text-neutral-500" />}
                  defaultExpanded={true}
                >
                  <div className="p-4">
                    <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Query Time Breakdown</h3>
                    
                    {/* Time Breakdown Chart */}
                    <div className="space-y-2 mt-4">
                      {[
                        { label: 'SQL Generation', value: data.performance_metrics.avg_sql_generation_time_ms, color: 'rgba(59, 130, 246, 0.8)' },
                        { label: 'SQL Execution', value: data.performance_metrics.avg_sql_execution_time_ms, color: 'rgba(16, 185, 129, 0.8)' },
                        { label: 'Visualization', value: data.performance_metrics.avg_visualization_time_ms, color: 'rgba(245, 158, 11, 0.8)' },
                        { label: 'Explanation', value: data.performance_metrics.avg_explanation_time_ms, color: 'rgba(139, 92, 246, 0.8)' }
                      ].map((item, index) => {
                        const totalTime = data.performance_metrics.avg_total_time_ms;
                        const percentage = totalTime > 0 ? (item.value / totalTime) * 100 : 0;
                        
                        return (
                          <div key={index} className="flex items-center">
                            <div className="w-32 text-xs text-neutral-600 dark:text-neutral-400 truncate mr-2">{item.label}</div>
                            <div className="flex-1 h-5 bg-neutral-100 dark:bg-neutral-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full rounded-full" 
                                style={{ 
                                  width: `${percentage}%`,
                                  backgroundColor: item.color
                                }}
                              ></div>
                            </div>
                            <div className="ml-2 text-xs text-neutral-600 dark:text-neutral-400 min-w-[80px] text-right">
                              {formatTime(item.value)} ({percentage.toFixed(1)}%)
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    
                    {/* Time series chart for performance over time */}
                    {data.time_series.dates.length > 0 && (
                      <div className="mt-6">
                        <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Performance Over Time</h3>
                        {renderLineChart(
                          data.time_series.dates,
                          [
                            { 
                              label: 'Avg Query Time (ms)', 
                              data: data.time_series.avg_times,
                              color: 'rgba(59, 130, 246, 0.8)'
                            }
                          ]
                        )}
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
              </div>
              
              {/* Success Rate and Retries Section */}
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden">
                <CollapsibleSection 
                  title="Success Rate & Retries" 
                  icon={<CheckCircle className="w-5 h-5 text-neutral-500" />}
                  defaultExpanded={true}
                >
                  <div className="p-4">
                    {/* Success/Error Pie Chart */}
                    <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Query Success Breakdown</h3>
                    {renderPieChart([
                      { label: 'Success (First Try)', value: data.successful_queries - data.retry_metrics.successful_after_retry, color: 'rgba(16, 185, 129, 0.8)' },
                      { label: 'Success (After Retry)', value: data.retry_metrics.successful_after_retry, color: 'rgba(245, 158, 11, 0.8)' },
                      { label: 'Failed', value: data.error_queries, color: 'rgba(239, 68, 68, 0.8)' }
                    ])}
                    
                    {/* Time series for success rate */}
                    {data.time_series.dates.length > 0 && (
                      <div className="mt-6">
                        <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Success Rate Over Time</h3>
                        {renderLineChart(
                          data.time_series.dates,
                          [
                            { 
                              label: 'Success Rate (%)', 
                              data: data.time_series.success_rates,
                              color: 'rgba(16, 185, 129, 0.8)'
                            },
                            { 
                              label: 'Retries', 
                              data: data.time_series.retries,
                              color: 'rgba(245, 158, 11, 0.8)'
                            }
                          ]
                        )}
                      </div>
                    )}
                    
                    {/* Retry statistics */}
                    <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Queries with Retries</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.retry_metrics.queries_with_retries)}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {((data.retry_metrics.queries_with_retries / data.total_queries) * 100).toFixed(1)}% of total
                        </div>
                      </div>
                      
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Total Retries</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.retry_metrics.total_retries)}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {formatNumber(data.retry_metrics.avg_retries)} avg per query
                        </div>
                      </div>
                      
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Retry Success Rate</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.retry_metrics.retry_success_rate)}%
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {formatNumber(data.retry_metrics.successful_after_retry)} fixed after retry
                        </div>
                      </div>
                    </div>
                  </div>
                </CollapsibleSection>
              </div>
              
              {/* Error Analysis Section */}
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden">
                <CollapsibleSection 
                  title="Error Analysis" 
                  icon={<XCircle className="w-5 h-5 text-neutral-500" />}
                  defaultExpanded={true}
                >
                  <div className="p-4">
                    <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Error Types</h3>
                    {data.error_analysis && data.error_analysis.length > 0 ? (
                      renderBarChart(data.error_analysis, 'Common Error Types', 'rgba(239, 68, 68, 0.8)')
                    ) : (
                      <div className="text-center p-4 bg-neutral-50 dark:bg-neutral-750 rounded-lg">
                        <p className="text-sm text-neutral-500 dark:text-neutral-400">No errors found in this time period</p>
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
              </div>
              
              {/* Query Patterns Section */}
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden">
                <CollapsibleSection 
                  title="Query Patterns" 
                  icon={<BarChart className="w-5 h-5 text-neutral-500" />}
                  defaultExpanded={true}
                >
                  <div className="p-4">
                    <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Query Types</h3>
                    {renderBarChart(data.query_pattern_analysis, 'Common Query Patterns', 'rgba(139, 92, 246, 0.8)')}
                    
                    {/* Query complexity metrics */}
                    <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Avg SQL Length</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.complexity_metrics.avg_sql_length)} chars
                        </div>
                      </div>
                      
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Avg Result Size</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.complexity_metrics.avg_result_rows)} rows
                        </div>
                      </div>
                    </div>
                    
                    {/* Time series for query volume */}
                    {data.time_series.dates.length > 0 && (
                      <div className="mt-6">
                        <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Query Volume Over Time</h3>
                        {renderLineChart(
                          data.time_series.dates,
                          [{ 
                            label: 'Queries', 
                            data: data.time_series.counts,
                            color: 'rgba(139, 92, 246, 0.8)'
                          }]
                        )}
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
              </div>
              
              {/* Visualization Metrics */}
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden lg:col-span-2">
                <CollapsibleSection 
                  title="Visualization Metrics" 
                  icon={<PieChart className="w-5 h-5 text-neutral-500" />}
                  defaultExpanded={true}
                >
                  <div className="p-4">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Visualization Rate</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatNumber(data.complexity_metrics.visualization_rate)}%
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          of successful queries
                        </div>
                      </div>
                      
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Avg Visualization Time</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatTime(data.performance_metrics.avg_visualization_time_ms)}
                        </div>
                      </div>
                      
                      <div className="bg-neutral-50 dark:bg-neutral-750 p-3 rounded-lg">
                        <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">Avg Explanation Time</div>
                        <div className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {formatTime(data.performance_metrics.avg_explanation_time_ms)}
                        </div>
                      </div>
                    </div>
                  </div>
                </CollapsibleSection>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 