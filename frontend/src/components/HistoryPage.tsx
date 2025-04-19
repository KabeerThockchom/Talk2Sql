import React, { useState } from 'react';
import { useQuery, useMutation, QueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { FileCode, ArrowLeft, Download, Table, BarChart, FileText, Database, Brain, Clock, Search, Loader2, RefreshCw, History as HistoryIcon, Filter } from 'lucide-react';
import { CollapsibleSection } from './CollapsibleSection';
import { PlotlyChart } from './PlotlyChart';
import { Tooltip } from './Tooltip';

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

interface HistoryItem {
  id: string;
  question: string;
  sql: string;
  timestamp: string;
  success: boolean;
  error?: string;
  data?: any[];
  columns?: string[];
  visualization?: string;
  summary?: string;
  used_memory?: boolean;
}

interface HistoryPageProps {
  onBack: () => void;
  queryClient: QueryClient;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ onBack, queryClient }) => {
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const [exportFormat, setExportFormat] = useState<'json' | 'csv' | 'full_csv'>('json');
  const [isExporting, setIsExporting] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [timeFilter, setTimeFilter] = useState<'all' | 'day' | 'week' | 'month'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'success' | 'failed'>('all');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['history'],
    queryFn: async () => {
      const response = await axios.get('https://text2sql.fly.dev/history');
      return response.data;
    }
  });

  // Add a mutation for rerunning queries
  const rerunMutation = useMutation({
    mutationFn: async (question: string) => {
      const response = await axios.post('https://text2sql.fly.dev/ask', { question });
      return response.data;
    },
    onSuccess: (data) => {
      // Store the result in query cache for the main view to pick up
      queryClient.setQueryData(['history-rerun'], data);
      // Navigate back to main view to show results
      onBack();
    }
  });

  const handleRerun = async (question: string) => {
    try {
      await rerunMutation.mutateAsync(question);
    } catch (error) {
      console.error('Error rerunning query:', error);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const handleDownloadCSV = (data: any[], columns: string[]) => {
    // Create CSV header row
    let csvContent = columns.join(',') + '\n';
    
    // Add data rows
    data.forEach(row => {
      const rowValues = columns.map(column => {
        // Handle values that might contain commas by enclosing in quotes
        const value = row[column] !== null && row[column] !== undefined ? row[column] : '';
        return typeof value === 'string' && value.includes(',') 
          ? `"${value}"` 
          : value;
      });
      csvContent += rowValues.join(',') + '\n';
    });
    
    // Create a blob and download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `query_results_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportHistory = async (format: 'json' | 'csv' | 'full_csv', itemId?: string) => {
    try {
      setIsExporting(true);
      
      // Construct the URL with format and optional ID
      let url = `https://text2sql.fly.dev/export_history?format=${format}`;
      if (itemId) {
        url += `&id=${itemId}`;
      }
      
      // For JSON format, use axios and save file manually
      if (format === 'json') {
        const response = await axios.get(url);
        
        // Convert to JSON string with nice formatting
        const jsonData = JSON.stringify(response.data, null, 2);
        
        // Create a blob and trigger download
        const blob = new Blob([jsonData], { type: 'application/json' });
        const downloadUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = itemId 
          ? `query_${itemId}_${new Date().toISOString().slice(0, 10)}.json` 
          : `history_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(downloadUrl);
      } else {
        // For CSV and full_csv formats, open in new window to trigger download
        window.open(url, '_blank');
      }
    } catch (error) {
      console.error('Error exporting history:', error);
    } finally {
      setIsExporting(false);
    }
  };

  // Filter history items based on search query and filters
  const filteredHistory = data?.history?.filter((item: HistoryItem) => {
    // Filter by search query
    if (searchQuery && !item.question.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    
    // Filter by status
    if (statusFilter === 'success' && !item.success) {
      return false;
    }
    if (statusFilter === 'failed' && item.success) {
      return false;
    }
    
    // Filter by time
    if (timeFilter !== 'all') {
      const now = new Date();
      const itemDate = new Date(item.timestamp);
      const diffDays = Math.floor((now.getTime() - itemDate.getTime()) / (1000 * 60 * 60 * 24));
      
      if (timeFilter === 'day' && diffDays > 1) {
        return false;
      }
      if (timeFilter === 'week' && diffDays > 7) {
        return false;
      }
      if (timeFilter === 'month' && diffDays > 30) {
        return false;
      }
    }
    
    return true;
  }) || [];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[280px] bg-white dark:bg-neutral-800 border-r border-neutral-200 dark:border-neutral-700 flex flex-col">
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center gap-2 mb-4">
            <HistoryIcon className="w-6 h-6 text-secondary" style={{ color: COLORS.bookCloth }} />
            <h1 className="text-lg font-semibold text-neutral-900 dark:text-white">Query History</h1>
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
          <div className="mb-4">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
              <input
                type="text"
                placeholder="Search queries..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-neutral-100 dark:bg-neutral-700 border border-neutral-200 dark:border-neutral-600 rounded-lg text-sm text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2">TIME RANGE</h2>
          <div className="space-y-1 mb-4">
            {[
              { label: 'All Time', value: 'all' },
              { label: 'Last 24 Hours', value: 'day' },
              { label: 'Last 7 Days', value: 'week' },
              { label: 'Last 30 Days', value: 'month' }
            ].map(option => (
              <button
                key={option.value}
                onClick={() => setTimeFilter(option.value as any)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                  timeFilter === option.value
                    ? 'bg-primary text-white'
                    : 'text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                }`}
                style={{ 
                  backgroundColor: timeFilter === option.value 
                    ? COLORS.bookCloth 
                    : 'transparent'
                }}
              >
                <Clock className="w-4 h-4" />
                <span>{option.label}</span>
              </button>
            ))}
          </div>

          <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2">STATUS</h2>
          <div className="space-y-1 mb-4">
            {[
              { label: 'All Queries', value: 'all' },
              { label: 'Successful', value: 'success' },
              { label: 'Failed', value: 'failed' }
            ].map(option => (
              <button
                key={option.value}
                onClick={() => setStatusFilter(option.value as any)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                  statusFilter === option.value
                    ? 'bg-primary text-white'
                    : 'text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                }`}
                style={{ 
                  backgroundColor: statusFilter === option.value 
                    ? COLORS.kraft 
                    : 'transparent'
                }}
              >
                <Filter className="w-4 h-4" />
                <span>{option.label}</span>
              </button>
            ))}
          </div>

          <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2">EXPORT OPTIONS</h2>
          <div className="bg-neutral-100 dark:bg-neutral-700 rounded-lg p-3 mb-4">
            <select 
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value as 'json' | 'csv' | 'full_csv')}
              className="w-full mb-2 bg-white dark:bg-neutral-600 border border-neutral-300 dark:border-neutral-600 text-sm rounded-md px-2 py-1"
            >
              <option value="json">JSON Format</option>
              <option value="csv">CSV Summary</option>
              <option value="full_csv">Full Export (ZIP)</option>
            </select>
            <button
              onClick={() => handleExportHistory(exportFormat)}
              disabled={isExporting}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-white dark:bg-neutral-600 rounded-lg text-neutral-700 dark:text-neutral-200 hover:bg-neutral-200 dark:hover:bg-neutral-500 transition-colors"
            >
              <Download className="w-4 h-4" style={{ color: COLORS.bookCloth }} />
              <span>{isExporting ? 'Exporting...' : 'Export All'}</span>
            </button>
          </div>
        </div>

        <div className="p-4 border-t border-neutral-200 dark:border-neutral-700">
          <button
            onClick={() => refetch()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg text-neutral-600 dark:text-neutral-200 hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors"
          >
            <RefreshCw className="w-4 h-4" style={{ color: COLORS.bookCloth }} />
            <span>Refresh History</span>
          </button>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex-1 overflow-auto p-4 bg-neutral-50 dark:bg-neutral-900">
        <div className="max-w-6xl mx-auto">
          <div className="mb-4">
            <h1 className="text-xl font-semibold text-neutral-900 dark:text-white">Query History</h1>
            <p className="text-sm text-neutral-500 dark:text-neutral-400">
              {filteredHistory.length} {filteredHistory.length === 1 ? 'query' : 'queries'} found
            </p>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center h-64 bg-white dark:bg-neutral-800 rounded-lg shadow-sm">
              <Loader2 className="w-6 h-6 animate-spin text-primary" style={{ color: COLORS.bookCloth }} />
              <p className="ml-2 text-neutral-600 dark:text-neutral-300">Loading history...</p>
            </div>
          ) : error ? (
            <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-6 border-l-4 border-red-500">
              <h3 className="text-red-500 font-medium mb-2">Error</h3>
              <p className="text-neutral-700 dark:text-neutral-300">
                Failed to load history. Please try again.
              </p>
            </div>
          ) : (
            <>
              {filteredHistory.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {filteredHistory.map((item: HistoryItem) => (
                    <div 
                      key={item.id} 
                      className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm overflow-hidden h-full flex flex-col border border-neutral-200 dark:border-neutral-700"
                    >
                      <div 
                        className="p-4 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-750 transition-colors"
                        onClick={() => setSelectedItem(selectedItem?.id === item.id ? null : item)}
                      >
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <p className="font-medium text-neutral-900 dark:text-white mb-1 line-clamp-2">{item.question}</p>
                            <div className="flex items-center text-sm text-neutral-500 dark:text-neutral-400 mb-2">
                              <Clock className="w-3 h-3 mr-1" />
                              {formatTimestamp(item.timestamp)}
                            </div>
                            
                            {/* Quick info badges */}
                            <div className="flex flex-wrap gap-2 mt-2">
                              {item.data && (
                                <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-xs rounded-full flex items-center">
                                  <Table className="w-3 h-3 mr-1" /> 
                                  {item.data.length} rows
                                </span>
                              )}
                              {item.summary && (
                                <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200 text-xs rounded-full flex items-center">
                                  <FileText className="w-3 h-3 mr-1" />
                                  Summary
                                </span>
                              )}
                              {item.visualization && (
                                <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 text-xs rounded-full flex items-center">
                                  <BarChart className="w-3 h-3 mr-1" />
                                  Chart
                                </span>
                              )}
                              {item.used_memory !== undefined && (
                                <span className={`px-2 py-1 ${item.used_memory ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200' : 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-200'} text-xs rounded-full flex items-center`}>
                                  <Brain className="w-3 h-3 mr-1" />
                                  {item.used_memory ? 'Memory Used' : 'No Memory'}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex flex-col items-end gap-2 ml-2">
                            {item.success ? (
                              <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 text-xs rounded-full">
                                Success
                              </span>
                            ) : (
                              <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200 text-xs rounded-full">
                                Failed
                              </span>
                            )}
                            <div className="flex gap-1">
                              <Tooltip content="Run this query again" offset={24}>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleRerun(item.question);
                                  }}
                                  disabled={rerunMutation.isPending}
                                  className="p-1.5 rounded-md text-neutral-500 hover:text-primary hover:bg-primary/10 dark:hover:bg-neutral-600 disabled:opacity-50"
                                  style={{ color: rerunMutation.isPending ? undefined : COLORS.bookCloth }}
                                >
                                  <RefreshCw className="w-4 h-4" />
                                </button>
                              </Tooltip>
                              <Tooltip content="Export query data" position="bottom" offset={24}>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleExportHistory(exportFormat, item.id);
                                  }}
                                  disabled={isExporting}
                                  className="p-1.5 rounded-md text-neutral-500 hover:text-primary hover:bg-primary/10 dark:hover:bg-neutral-600 disabled:opacity-50"
                                  style={{ color: isExporting ? undefined : COLORS.kraft }}
                                >
                                  <Download className="w-4 h-4" />
                                </button>
                              </Tooltip>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* SQL Preview */}
                      {selectedItem?.id !== item.id && item.sql && (
                        <div className="mt-2 p-2 bg-neutral-50 dark:bg-neutral-750 rounded border border-neutral-200 dark:border-neutral-700 mx-4 mb-4">
                          <div className="flex items-center mb-1">
                            <FileCode className="w-3 h-3 text-secondary mr-1" style={{ color: COLORS.bookCloth }} />
                            <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">SQL</span>
                          </div>
                          <pre className="text-xs text-neutral-800 dark:text-neutral-200 line-clamp-2 font-mono overflow-x-auto">{item.sql}</pre>
                        </div>
                      )}
                      
                      {/* Results Preview (if not expanded) */}
                      {selectedItem?.id !== item.id && item.data && item.columns && item.data.length > 0 && (
                        <div className="mt-2 p-2 bg-neutral-50 dark:bg-neutral-750 rounded border border-neutral-200 dark:border-neutral-700 mx-4 mb-4">
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center">
                              <Table className="w-3 h-3 text-secondary mr-1" style={{ color: COLORS.kraft }} />
                              <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Results Preview</span>
                            </div>
                            <Tooltip content="Download results as CSV" position="bottom" offset={24}>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  item.data && item.columns && handleDownloadCSV(item.data, item.columns);
                                }}
                                className="p-1 rounded-md text-neutral-500 hover:text-primary hover:bg-primary/10"
                              >
                                <Download className="w-3 h-3" />
                              </button>
                            </Tooltip>
                          </div>
                          <div className="overflow-x-auto max-h-[100px]">
                            <table className="min-w-full text-xs">
                              <thead>
                                <tr className="bg-neutral-100 dark:bg-neutral-700">
                                  {item.columns.slice(0, 3).map((column, idx) => (
                                    <th key={idx} className="px-2 py-1 text-left font-medium text-neutral-500 dark:text-neutral-400">
                                      {column}
                                    </th>
                                  ))}
                                  {item.columns.length > 3 && (
                                    <th className="px-2 py-1 text-left font-medium text-neutral-500 dark:text-neutral-400">...</th>
                                  )}
                                </tr>
                              </thead>
                              <tbody>
                                {item.data.slice(0, 2).map((row: any, rowIdx: number) => (
                                  <tr key={rowIdx} className="border-t border-neutral-200 dark:border-neutral-700">
                                    {item.columns!.slice(0, 3).map((column, colIdx) => (
                                      <td key={colIdx} className="px-2 py-1 text-neutral-900 dark:text-neutral-100 truncate max-w-[100px]">
                                        {row[column]}
                                      </td>
                                    ))}
                                    {item.columns!.length > 3 && (
                                      <td className="px-2 py-1 text-neutral-900 dark:text-neutral-100">...</td>
                                    )}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                      
                      {/* Expanded view */}
                      {selectedItem?.id === item.id && (
                        <div className="border-t border-neutral-200 dark:border-neutral-700 p-4 flex-1 overflow-auto max-h-[500px]">
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            {/* Summary Section - if available */}
                            {item.summary && (
                              <CollapsibleSection 
                                title="Summary" 
                                icon={<FileText className="w-4 h-4 text-neutral-500" />}
                                defaultExpanded={true}
                                className="mb-4 lg:col-span-2"
                              >
                                <div className="text-neutral-800 dark:text-neutral-200">
                                  {item.summary}
                                </div>
                              </CollapsibleSection>
                            )}
                            
                            {/* SQL Query Section */}
                            <CollapsibleSection 
                              title="Generated SQL" 
                              icon={<FileCode className="w-4 h-4 text-neutral-500" />}
                              defaultExpanded={true}
                              className="mb-4"
                            >
                              <pre className="bg-neutral-100 dark:bg-neutral-700 rounded-lg p-4 text-sm font-mono overflow-x-auto max-h-[200px]">
                                {item.sql}
                              </pre>
                            </CollapsibleSection>
                            
                            {/* Data Results Section - if available */}
                            {item.data && item.columns && (
                              <CollapsibleSection 
                                title="Results" 
                                icon={<Table className="w-4 h-4 text-secondary" />}
                                defaultExpanded={true}
                                className="mb-4"
                              >
                                <div className="mb-2 flex justify-end">
                                  <Tooltip content="Download results as CSV" position="bottom" offset={24}>
                                    <button
                                      onClick={() => item.data && item.columns && handleDownloadCSV(item.data, item.columns)}
                                      className="flex items-center gap-1 px-3 py-1.5 text-white rounded-md hover:bg-opacity-90 transition-colors text-xs"
                                      style={{ backgroundColor: COLORS.bookCloth }}
                                    >
                                      <Download className="w-3 h-3" />
                                      <span>Download CSV</span>
                                    </button>
                                  </Tooltip>
                                </div>
                                <div className="overflow-x-auto max-h-[300px] border border-neutral-200 dark:border-neutral-700 rounded-lg">
                                  {item.data && item.columns && item.data.length > 0 && item.columns.length > 0 && (
                                    <table className="min-w-full divide-y divide-neutral-200 dark:divide-neutral-700">
                                      <thead className="bg-neutral-50 dark:bg-neutral-800 sticky top-0">
                                        <tr>
                                          {item.columns.map((column, idx) => (
                                            <th key={idx} className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                                              {column}
                                            </th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody className="divide-y divide-neutral-200 dark:divide-neutral-700 bg-white dark:bg-neutral-700">
                                        {/* Show all rows up to 100, with a message if there are more */}
                                        {item.data.slice(0, 100).map((row: any, rowIdx: number) => (
                                          <tr key={rowIdx}>
                                            {item.columns!.map((column, colIdx) => (
                                              <td key={colIdx} className="px-3 py-2 text-xs text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                                                {row[column]}
                                              </td>
                                            ))}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  )}
                                  {item.data && item.data.length > 100 && (
                                    <div className="text-center p-2 text-xs text-neutral-500 dark:text-neutral-400 bg-neutral-50 dark:bg-neutral-800 border-t border-neutral-200 dark:border-neutral-700">
                                      Showing 100 of {item.data.length} rows. Download CSV for complete data.
                                    </div>
                                  )}
                                </div>
                              </CollapsibleSection>
                            )}
                            
                            {/* Visualization Section - if available */}
                            {item.visualization && (
                              <CollapsibleSection 
                                title="Visualization" 
                                icon={<BarChart className="w-4 h-4 text-neutral-500" />}
                                defaultExpanded={true}
                                className="mb-4 lg:col-span-2"
                              >
                                <div className="h-[300px]">
                                  <PlotlyChart visualizationJson={item.visualization} />
                                </div>
                              </CollapsibleSection>
                            )}
                          </div>
                          
                          {item.error && (
                            <div className="mt-4 p-3 bg-red-50 dark:bg-red-900 dark:bg-opacity-20 border border-red-200 dark:border-red-800 rounded-lg">
                              <h4 className="text-sm font-medium text-red-800 dark:text-red-300 mb-1">Error</h4>
                              <p className="text-sm text-red-700 dark:text-red-400">{item.error}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-6 text-center">
                  <p className="text-neutral-600 dark:text-neutral-300">
                    {searchQuery || timeFilter !== 'all' || statusFilter !== 'all' 
                      ? 'No queries match your current filters.' 
                      : 'No query history found.'}
                  </p>
                  {(searchQuery || timeFilter !== 'all' || statusFilter !== 'all') && (
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setTimeFilter('all');
                        setStatusFilter('all');
                      }}
                      className="mt-2 px-4 py-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-600 text-sm"
                    >
                      Clear all filters
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}; 