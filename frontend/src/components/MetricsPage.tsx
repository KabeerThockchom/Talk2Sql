import React, { useState, ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { ArrowLeft, BarChart, Clock, RefreshCw, Calendar, CheckCircle, XCircle, Database, LineChart as LineChartIcon, PieChart as PieChartIcon, Brain, Activity, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { ResponsiveContainer, Line, XAxis, YAxis, LineChart, PieChart, Pie, Cell, Legend, Tooltip as RechartsTooltip, BarChart as RechartsBarChart, Bar } from 'recharts';

// Define brand colors as constants
const COLORS = {
  // Primary
  bookCloth: '#CC785C',
  kraft: '#D4A27F',
  manilla: '#EBDBBC',
  
  // Secondary
  slateDark: '#191919',
  slateMedium: '#262625',
  slateLight: '#40403E',
  
  // Background
  white: '#FFFFFF',
  ivoryMedium: '#F0F0EB',
  ivoryLight: '#FAFAF7',
};

// CollapsibleSection component
interface CollapsibleSectionProps {
  title: string;
  children: ReactNode;
  defaultExpanded?: boolean;
  icon?: ReactNode;
  className?: string;
}

function CollapsibleSection({
  title,
  children,
  defaultExpanded = true,
  icon,
  className = '',
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const toggleExpanded = () => setIsExpanded(prev => !prev);

  return (
    <div style={{ backgroundColor: COLORS.white }} className={`rounded-lg shadow-sm overflow-hidden ${className}`}>
      <button
        onClick={toggleExpanded}
        className="w-full px-4 py-3 flex items-center justify-between hover:opacity-80 transition-opacity"
        style={{ color: COLORS.slateDark }}
      >
        <div className="flex items-center gap-2">
          {icon && <div style={{ color: COLORS.slateMedium }}>{icon}</div>}
          <h3 className="text-sm font-medium">{title}</h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4" style={{ color: COLORS.slateMedium }} />
        ) : (
          <ChevronDown className="w-4 h-4" style={{ color: COLORS.slateMedium }} />
        )}
      </button>
      
      {isExpanded && (
        <div className="px-4 pb-4" style={{ borderTopColor: COLORS.ivoryMedium }}>
          {children}
        </div>
      )}
    </div>
  );
}

interface MetricsPageProps {
  onBack: () => void;
}

interface MetricsResponse {
  status?: string;
  message?: string;
  total_queries: number;
  successful_queries: number;
  error_queries: number;
  success_rate: number;
  
  latency: {
    p50_total_ms: number;
    p95_total_ms: number;
    stage_p95_ms: {
      generation_ms: number;
      execution_ms: number;
      visualization_ms: number;
      explanation_ms: number;
    };
    mean_breakdown_pct: {
      generation_pct: number;
      execution_pct: number;
      visualization_pct: number;
      explanation_pct: number;
    };
  };
  
  retry_metrics: {
    queries_with_retry: number;
    total_retries: number;
    retry_rate_pct: number;
    retry_success_rate_pct: number;
  };
  
  memory_metrics: {
    queries_with_memory: number;
    memory_usage_rate_pct: number;
    with_memory_success_rate_pct: number;
    without_memory_success_rate_pct: number;
  };
  
  top_errors: Array<{type: string, count: number}>;
  
  time_series: {
    dates: string[];
    counts: number[];
    success_counts: number[];
    success_rates: number[];
    retries: number[];
  };
}

export const MetricsPage: React.FC<MetricsPageProps> = ({ onBack }) => {
  const [timeRange, setTimeRange] = useState<'all' | 'day' | 'week' | 'month'>('all');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['metrics', timeRange],
    queryFn: async () => {
      const response = await axios.get<MetricsResponse>(`https://text2sql.fly.dev/metrics?time_range=${timeRange}`);
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

  // Add formatDate function if it doesn't exist
  const formatDate = (dateStr: string): string => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // Metric Card component for consistent styling
  const MetricCard = ({ 
    title, 
    value, 
    subtitle = null, 
    icon = null, 
    valueColor = COLORS.slateDark
  }: { 
    title: string, 
    value: string | number | React.ReactNode, 
    subtitle?: string | null, 
    icon?: React.ReactNode | null,
    valueColor?: string
  }) => (
    <div style={{ backgroundColor: COLORS.white }} className="rounded-lg shadow-sm p-4">
      <div className="flex items-center mb-1">
        {icon && <div className="w-5 h-5 mr-2" style={{ color: valueColor }}>{icon}</div>}
        <h3 className="text-xs font-medium" style={{ color: COLORS.slateMedium }}>{title}</h3>
      </div>
      <div className="text-3xl font-bold" style={{ color: valueColor }}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs mt-1" style={{ color: COLORS.slateMedium }}>
          {subtitle}
        </div>
      )}
    </div>
  );

  // Component for latency breakdown
  const LatencyBreakdownChart = ({ breakdownData }: { 
    breakdownData: {
      generation_pct: number;
      execution_pct: number;
      visualization_pct: number;
      explanation_pct: number;
    }
  }) => {
    const data = [
      { name: 'SQL Generation', value: breakdownData.generation_pct, color: COLORS.bookCloth },
      { name: 'SQL Execution', value: breakdownData.execution_pct, color: COLORS.kraft },
      { name: 'Visualization', value: breakdownData.visualization_pct, color: COLORS.manilla },
      { name: 'Explanation', value: breakdownData.explanation_pct, color: COLORS.slateMedium }
    ];
    
    return (
      <div className="space-y-1 mt-2">
        {data.map((item, index) => (
          <div key={index} className="flex items-center">
            <div className="w-24 text-xs truncate mr-1" style={{ color: COLORS.slateMedium }}>{item.name}</div>
            <div className="flex-1 h-4 rounded-full overflow-hidden" style={{ backgroundColor: COLORS.ivoryMedium }}>
              <div 
                className="h-full rounded-full" 
                style={{ 
                  width: `${item.value}%`,
                  backgroundColor: item.color
                }}
              ></div>
            </div>
            <div className="ml-1 text-xs min-w-[40px] text-right" style={{ color: COLORS.slateMedium }}>
              {item.value.toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Component for rendering the P95 latency by stage
  const StageLatencyChart = ({ stageData }: {
    stageData: {
      generation_ms: number;
      execution_ms: number;
      visualization_ms: number;
      explanation_ms: number;
    }
  }) => {
    const chartData = [
      { name: 'Generation', value: stageData.generation_ms },
      { name: 'Execution', value: stageData.execution_ms },
      { name: 'Visualization', value: stageData.visualization_ms },
      { name: 'Explanation', value: stageData.explanation_ms }
    ];

    return (
      <div className="w-full h-36 mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            barSize={10}
          >
            <XAxis 
              type="number" 
              tick={{ fontSize: 10, fill: COLORS.slateMedium }} 
            />
            <YAxis 
              type="category" 
              dataKey="name" 
              tick={{ fontSize: 10, fill: COLORS.slateMedium }} 
              width={80} 
            />
            <RechartsTooltip
              formatter={(value: number) => formatTime(value)}
              labelFormatter={(label) => `P95 ${label} Latency`}
              contentStyle={{ backgroundColor: COLORS.white, borderColor: COLORS.ivoryMedium }}
            />
            <Bar dataKey="value" radius={4}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={
                  index === 0 ? COLORS.bookCloth :
                  index === 1 ? COLORS.kraft :
                  index === 2 ? COLORS.manilla :
                  COLORS.slateMedium
                } />
              ))}
            </Bar>
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // For the error bar chart
  const renderBarChart = (data: { type: string, count: number }[], maxBars: number = 5) => {
    if (!data || data.length === 0) return null;
    
    // Sort by count descending
    const sortedData = [...data].sort((a, b) => b.count - a.count).slice(0, maxBars);
    
    // Get max value for scaling
    const maxCount = Math.max(...sortedData.map(item => item.count));
    
    return (
      <div className="space-y-1 mt-2">
        {sortedData.map((item, index) => (
          <div key={index} className="flex items-center">
            <div className="w-24 text-xs truncate mr-1" style={{ color: COLORS.slateMedium }}>{item.type}</div>
            <div className="flex-1 h-4 rounded-full overflow-hidden" style={{ backgroundColor: COLORS.ivoryMedium }}>
              <div 
                className="h-full rounded-full" 
                style={{ 
                  width: `${(item.count / maxCount) * 100}%`,
                  backgroundColor: COLORS.bookCloth
                }}
              ></div>
            </div>
            <div className="ml-1 text-xs min-w-8 text-right" style={{ color: COLORS.slateMedium }}>{item.count}</div>
          </div>
        ))}
      </div>
    );
  };

  // Memory usage chart component
  const MemoryUsageChart = () => {
    if (!data || !data.memory_metrics) return null;
    
    const memoryData = [
      { 
        name: 'With Memory', 
        value: data.memory_metrics.queries_with_memory,
        successRate: data.memory_metrics.with_memory_success_rate_pct 
      },
      { 
        name: 'Without Memory', 
        value: data.total_queries - data.memory_metrics.queries_with_memory,
        successRate: data.memory_metrics.without_memory_success_rate_pct 
      }
    ];
    
    const chartData = [
      { name: 'With Memory', value: data.memory_metrics.queries_with_memory, fill: COLORS.kraft },
      { name: 'Without Memory', value: data.total_queries - data.memory_metrics.queries_with_memory, fill: COLORS.bookCloth }
    ];

    // Calculate percentages for display
    const withMemoryPct = data.memory_metrics.queries_with_memory / data.total_queries * 100;
    const withoutMemoryPct = 100 - withMemoryPct;
    
    return (
      <div className="flex flex-col">
        <div className="w-full h-36">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={50}
                fill="#8884d8"
                paddingAngle={5}
                dataKey="value"
                labelLine={false}
                label={false}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <RechartsTooltip
                content={(props: any) => {
                  const { active, payload } = props;
                  if (active && payload && payload.length && payload[0] && payload[0].name) {
                    const item = memoryData.find(d => d.name === payload[0].name);
                    return (
                      <div style={{ backgroundColor: COLORS.white, borderColor: COLORS.ivoryMedium }} className="p-2 text-xs border rounded shadow">
                        <p className="font-medium" style={{ color: COLORS.slateDark }}>{payload[0].name}</p>
                        <p style={{ color: COLORS.slateMedium }}>Queries: {payload[0].value}</p>
                        <p style={{ color: COLORS.slateMedium }}>Success Rate: {item ? item.successRate.toFixed(1) : 0}%</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex justify-center gap-4 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: COLORS.kraft }}></div>
            <span style={{ color: COLORS.slateMedium }}>With Memory {withMemoryPct.toFixed(0)}%</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: COLORS.bookCloth }}></div>
            <span style={{ color: COLORS.slateMedium }}>Without Memory {withoutMemoryPct.toFixed(0)}%</span>
          </div>
        </div>

        <div className="w-full h-4 bg-gray-200 rounded-full mt-2 overflow-hidden">
          <div className="flex h-full">
            <div 
              style={{ 
                width: `${withoutMemoryPct}%`, 
                backgroundColor: COLORS.bookCloth 
              }} 
              className="h-full"
            ></div>
            <div 
              style={{ 
                width: `${withMemoryPct}%`, 
                backgroundColor: COLORS.kraft 
              }} 
              className="h-full"
            ></div>
          </div>
        </div>
      </div>
    );
  };

  // Retry metrics component
  const RetryMetricsDisplay = () => {
    if (!data || !data.retry_metrics) return null;
    
    const retryRateColor = data.retry_metrics.retry_rate_pct > 50 ? COLORS.bookCloth : COLORS.slateMedium;
    
    return (
      <div className="flex flex-col gap-2 mt-2">
        <div className="flex items-center justify-between">
          <span style={{ color: COLORS.slateMedium }}>Retry Rate:</span>
          <span className="font-medium" style={{ color: retryRateColor }}>
            {data.retry_metrics.retry_rate_pct.toFixed(1)}%
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span style={{ color: COLORS.slateMedium }}>Queries with Retry:</span>
          <span className="font-medium" style={{ color: COLORS.slateDark }}>
            {data.retry_metrics.queries_with_retry} ({((data.retry_metrics.queries_with_retry / data.total_queries) * 100).toFixed(1)}%)
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span style={{ color: COLORS.slateMedium }}>Success After Retry:</span>
          <span className="font-medium" style={{ color: COLORS.slateDark }}>
            {data.retry_metrics.retry_success_rate_pct.toFixed(1)}%
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span style={{ color: COLORS.slateMedium }}>Total Retries:</span>
          <span className="font-medium" style={{ color: COLORS.slateDark }}>
            {data.retry_metrics.total_retries}
          </span>
        </div>
      </div>
    );
  };

  // For the dual-axis chart (counts and success rates)
  const renderPerformanceChart = () => {
    if (!data || !data.time_series || !data.time_series.dates) return null;

    const chartData = data.time_series.dates.map((date, index) => ({
      date,
      count: data.time_series.counts[index],
      successRate: data.time_series.success_rates[index]
    }));

    return (
      <div className="w-full h-48 mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
            barSize={20}
          >
            <XAxis 
              dataKey="date" 
              tickFormatter={formatDate} 
              tick={{ fill: COLORS.slateMedium }} 
            />
            <YAxis 
              yAxisId="left"
              orientation="left"
              tick={{ fontSize: 10, fill: COLORS.slateMedium }}
              label={{ value: 'Queries', angle: -90, position: 'insideLeft', fontSize: 10, fill: COLORS.slateMedium }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tickFormatter={(value) => `${value}%`}
              tick={{ fontSize: 10, fill: COLORS.slateMedium }}
              label={{ value: 'Success Rate', angle: 90, position: 'insideRight', fontSize: 10, fill: COLORS.slateMedium }}
              domain={[0, 100]}
            />
            <RechartsTooltip
              content={(props: any) => {
                const { active, payload, label } = props;
                if (active && payload && payload.length) {
                  return (
                    <div style={{ backgroundColor: COLORS.white, borderColor: COLORS.ivoryMedium }} className="p-2 text-xs border rounded shadow">
                      <p className="font-medium" style={{ color: COLORS.slateDark }}>{formatDate(label)}</p>
                      <p style={{ color: COLORS.slateMedium }}>Queries: {payload[0] && payload[0].value}</p>
                      <p style={{ color: COLORS.slateMedium }}>Success Rate: {payload[1] && payload[1].value ? payload[1].value.toFixed(1) : '0'}%</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar 
              yAxisId="left" 
              dataKey="count" 
              fill={COLORS.bookCloth} 
              radius={[4, 4, 0, 0]}
              name="Queries"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="successRate"
              stroke={COLORS.kraft}
              name="Success Rate"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Component for displaying custom Calendar metric
  const CalendarMetric = ({ dates, counts }: { dates: string[], counts: number[] }) => {
    const today = new Date();
    const firstDayOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
    const daysInMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
    
    // Create data map for lookup
    const dateMap = new Map();
    dates.forEach((date, index) => {
      dateMap.set(date, counts[index]);
    });
    
    // Create grid of days
    const days = [];
    for (let i = 1; i <= daysInMonth; i++) {
      const date = new Date(today.getFullYear(), today.getMonth(), i);
      const dateStr = date.toISOString().split('T')[0];
      const count = dateMap.get(dateStr) || 0;
      days.push({ date: i, count });
    }
    
    return (
      <div className="mt-4">
        <div className="flex items-center mb-2">
          <Calendar className="w-4 h-4 mr-2" style={{ color: COLORS.slateMedium }} />
          <span className="text-sm font-medium" style={{ color: COLORS.slateDark }}>Activity This Month</span>
        </div>
        <div className="grid grid-cols-7 gap-1">
          {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((day, i) => (
            <div key={i} className="text-center text-xs" style={{ color: COLORS.slateMedium }}>
              {day}
            </div>
          ))}
          
          {Array(firstDayOfMonth.getDay()).fill(null).map((_, i) => (
            <div key={`empty-${i}`} className="h-7"></div>
          ))}
          
          {days.map((day) => {
            const intensity = day.count === 0 ? 0 : Math.min(Math.ceil(day.count / 5), 4);
            const bgColor = intensity === 0 ? COLORS.ivoryMedium : 
                          intensity === 1 ? COLORS.manilla :
                          intensity === 2 ? COLORS.kraft :
                          intensity === 3 ? COLORS.bookCloth :
                          COLORS.slateDark;
            
            return (
              <div 
                key={day.date} 
                className="h-7 flex items-center justify-center rounded-sm"
                style={{ 
                  backgroundColor: bgColor,
                  color: intensity >= 3 ? COLORS.white : COLORS.slateDark
                }}
              >
                <span className="text-xs">{day.date}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Component for showing comparison chart with Legend
  const ComparisonChart = ({ successData, errorData }: { 
    successData: number[],
    errorData: number[] 
  }) => {
    const data = successData.map((val, idx) => ({
      name: `Day ${idx + 1}`,
      success: val,
      error: errorData[idx] || 0
    }));

    return (
      <div className="w-full h-48 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis dataKey="name" />
            <YAxis />
            <RechartsTooltip 
              content={(props: any) => {
                const { active, payload, label } = props;
                if (active && payload && payload.length) {
                  return (
                    <div style={{ backgroundColor: COLORS.white, borderColor: COLORS.ivoryMedium }} className="p-2 text-xs border rounded shadow">
                      <p className="font-medium" style={{ color: COLORS.slateDark }}>{label}</p>
                      <p style={{ color: COLORS.slateMedium }}>Success: {payload[0] && payload[0].value ? payload[0].value : 0}</p>
                      <p style={{ color: COLORS.slateMedium }}>Error: {payload[1] && payload[1].value ? payload[1].value : 0}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend 
              verticalAlign="top"
              height={36}
              formatter={(value) => <span style={{ color: value === 'success' ? COLORS.bookCloth : COLORS.slateLight }}>{value}</span>}
            />
            <Bar dataKey="success" stackId="a" fill={COLORS.bookCloth} />
            <Bar dataKey="error" stackId="a" fill={COLORS.slateLight} />
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="flex h-screen overflow-hidden" style={{ fontFamily: 'Styrene A, sans-serif', backgroundColor: COLORS.ivoryLight }}>
      {/* Sidebar */}
      <aside className="w-[280px] bg-white dark:bg-neutral-800 border-r border-neutral-200 dark:border-neutral-700 flex flex-col">
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center gap-2 mb-4">
            <Database className="w-6 h-6 text-secondary" style={{ color: COLORS.bookCloth }} />
            <h1 className="text-lg font-semibold text-neutral-900 dark:text-white">Analytics</h1>
          </div>
          <button
            onClick={onBack}
            className="w-full flex items-center gap-2 px-4 py-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg text-neutral-600 dark:text-neutral-200 hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to App</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2">TIME RANGE</h2>
          <div className="space-y-1">
            {[
              { label: 'All Time', value: 'all' },
              { label: 'Last 24 Hours', value: 'day' },
              { label: 'Last 7 Days', value: 'week' },
              { label: 'Last 30 Days', value: 'month' }
            ].map(option => (
              <button
                key={option.value}
                onClick={() => setTimeRange(option.value as any)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                  timeRange === option.value
                    ? 'bg-primary text-white'
                    : 'text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                }`}
                style={{ 
                  backgroundColor: timeRange === option.value 
                    ? COLORS.bookCloth 
                    : 'transparent'
                }}
              >
                <Calendar className="w-4 h-4" />
                <span>{option.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="p-4 border-t border-neutral-200 dark:border-neutral-700">
          <button
            onClick={() => refetch()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg text-neutral-600 dark:text-neutral-200 hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors"
          >
            <RefreshCw className="w-4 h-4" style={{ color: COLORS.bookCloth }} />
            <span>Refresh Data</span>
          </button>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex-1 overflow-auto p-4" style={{ backgroundColor: COLORS.ivoryLight }}>
        <div className="max-w-7xl mx-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-6 h-6 animate-spin" style={{ color: COLORS.slateMedium }} />
              <p className="ml-2" style={{ color: COLORS.slateMedium }}>Loading metrics...</p>
            </div>
          ) : error ? (
            <div className="rounded-lg shadow-sm p-6 border-l-4" style={{ backgroundColor: COLORS.white, borderColor: COLORS.bookCloth }}>
              <h3 className="font-medium mb-2" style={{ color: COLORS.bookCloth }}>Error</h3>
              <p style={{ color: COLORS.slateDark }}>
                Failed to load metrics. Please try again.
              </p>
            </div>
          ) : data?.status === "error" ? (
            <div className="rounded-lg shadow-sm p-6 border-l-4" style={{ backgroundColor: COLORS.white, borderColor: COLORS.kraft }}>
              <h3 className="font-medium mb-2" style={{ color: COLORS.kraft }}>No Data Available</h3>
              <p style={{ color: COLORS.slateDark }}>
                {data?.message || "No query history available for the selected time range."}
              </p>
            </div>
          ) : data ? (
            <>
              <div className="mb-4">
                <h1 className="text-lg font-semibold" style={{ color: COLORS.slateDark }}>Query Analytics Dashboard</h1>
                <p className="text-sm" style={{ color: COLORS.slateMedium }}>
                  {timeRange === 'all' ? 'All-time' : 
                  timeRange === 'day' ? 'Last 24 hours' :
                  timeRange === 'week' ? 'Last 7 days' : 'Last 30 days'} metrics based on {data.total_queries} queries
                </p>
              </div>
            
              {/* KPI row */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
                <MetricCard
                  title="Total Queries"
                  value={data.total_queries}
                  icon={<Database />}
                  valueColor={COLORS.bookCloth}
                />
                
                <MetricCard
                  title="Success Rate"
                  value={`${data.success_rate.toFixed(1)}%`}
                  subtitle={`${data.successful_queries} successful / ${data.error_queries} failed`}
                  icon={<CheckCircle />}
                  valueColor={COLORS.kraft}
                />
                
                <MetricCard
                  title="Latency (P50 / P95)"
                  value={`${formatTime(data.latency.p50_total_ms).split(' ')[0]} ${formatTime(data.latency.p50_total_ms).split(' ')[1]}`}
                  subtitle={`P95: ${formatTime(data.latency.p95_total_ms)}`}
                  icon={<Clock />}
                  valueColor={COLORS.manilla}
                />
                
                <MetricCard
                  title="Retry Stats"
                  value={`${data.retry_metrics.retry_rate_pct.toFixed(1)}%`}
                  subtitle={`Success after retry: ${data.retry_metrics.retry_success_rate_pct.toFixed(1)}%`}
                  icon={<RefreshCw />}
                  valueColor={COLORS.bookCloth}
                />
              </div>
              
              {/* Middle row: performance and errors */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                <CollapsibleSection 
                  title="Latency Breakdown" 
                  icon={<Clock className="w-4 h-4" />}
                >
                  <LatencyBreakdownChart breakdownData={data.latency.mean_breakdown_pct} />
                </CollapsibleSection>
                
                <CollapsibleSection 
                  title="Stage P95 Latency" 
                  icon={<Activity className="w-4 h-4" />}
                >
                  <StageLatencyChart stageData={data.latency.stage_p95_ms} />
                </CollapsibleSection>
              </div>
              
              {/* Bottom rows */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                <CollapsibleSection 
                  title="Error Analysis" 
                  icon={<XCircle className="w-4 h-4" />}
                >
                  <div className="mt-2">
                    <h3 className="text-sm font-medium" style={{ color: COLORS.slateDark }}>Top Error Types</h3>
                    {data.top_errors && data.top_errors.length > 0 ? (
                      renderBarChart(data.top_errors, 5)
                    ) : (
                      <div className="text-center p-2 rounded text-xs mt-2" style={{ backgroundColor: COLORS.ivoryMedium, color: COLORS.slateMedium }}>
                        No errors found in this time period
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
                
                <CollapsibleSection 
                  title="Retry Metrics" 
                  icon={<RefreshCw className="w-4 h-4" />}
                >
                  <RetryMetricsDisplay />
                </CollapsibleSection>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                <CollapsibleSection 
                  title="Memory Usage" 
                  icon={<Brain className="w-4 h-4" />}
                >
                  <MemoryUsageChart />
                </CollapsibleSection>
                
                <CollapsibleSection 
                  title="Activity Calendar" 
                  icon={<Calendar className="w-4 h-4" />}
                >
                  <CalendarMetric 
                    dates={data.time_series.dates}
                    counts={data.time_series.counts}
                  />
                </CollapsibleSection>
              </div>
              
              <div className="grid grid-cols-1 gap-3 mb-3">
                <CollapsibleSection 
                  title="Performance Over Time" 
                  icon={<LineChartIcon className="w-4 h-4" />}
                >
                  {renderPerformanceChart()}
                </CollapsibleSection>
              </div>
              
              <div className="grid grid-cols-1 gap-3">
                <CollapsibleSection 
                  title="Success vs Error Comparison" 
                  icon={<BarChart className="w-4 h-4" />}
                >
                  <ComparisonChart 
                    successData={data.time_series.success_counts}
                    errorData={data.time_series.counts.map((total, i) => 
                      total - (data.time_series.success_counts[i] || 0)
                    )}
                  />
                </CollapsibleSection>
                
                <CollapsibleSection 
                  title="Distribution Analysis" 
                  icon={<PieChartIcon className="w-4 h-4" />}
                >
                  <div className="w-full h-48 mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={data.time_series.dates.map((date, i) => ({
                          date,
                          success: data.time_series.success_rates[i],
                          retries: data.time_series.retries[i]
                        }))}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <XAxis dataKey="date" tickFormatter={formatDate} />
                        <YAxis />
                        <RechartsTooltip 
                          content={(props: any) => {
                            const { active, payload, label } = props;
                            if (active && payload && payload.length) {
                              return (
                                <div style={{ backgroundColor: COLORS.white, borderColor: COLORS.ivoryMedium }} className="p-2 text-xs border rounded shadow">
                                  <p className="font-medium" style={{ color: COLORS.slateDark }}>{formatDate(label)}</p>
                                  <p style={{ color: COLORS.slateMedium }}>Success Rate: {payload[0] && payload[0].value ? payload[0].value.toFixed(1) : '0'}%</p>
                                  <p style={{ color: COLORS.slateMedium }}>Retries: {payload[1] && payload[1].value ? payload[1].value : 0}</p>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="success" stroke={COLORS.bookCloth} name="Success Rate %" />
                        <Line type="monotone" dataKey="retries" stroke={COLORS.kraft} name="Retries" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CollapsibleSection>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
};