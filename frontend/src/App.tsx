import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { Database, History, Settings, Upload, Mic, Send, FileCode, Table, BarChart, FileText, Volume, ThumbsUp, ThumbsDown, Download, Activity, Loader } from 'lucide-react';
import { LoadingSteps, Step } from './components/LoadingSteps';
import { ThemeToggle } from './components/ThemeToggle';
import { CollapsibleSection } from './components/CollapsibleSection';
import { PlotlyChart } from './components/PlotlyChart';
import { AudioPlayer } from './components/AudioPlayer';
import { HistoryPage } from './components/HistoryPage';
import { MetricsPage } from './components/MetricsPage';
import { Tooltip } from './components/Tooltip';
import { FollowUpQuestions } from './components/FollowUpQuestion';

interface Database {
  name: string;
  path: string;
  has_persisted_vectors: boolean;
}

interface QueryResult {
  sql: string;
  data: any[];
  columns: string[];
  summary?: string;
  visualization?: string;
  audio_base64?: string;
  question: string;
}

// New interface for streaming query results
interface StreamingQueryResult {
  question: string;
  sql?: string;
  data?: any[];
  columns?: string[];
  summary?: string;
  visualization?: string;
  followups?: string[];
  isLoading: boolean;
  isLoadingSql: boolean;
  isLoadingData: boolean;
  isLoadingVisualization: boolean;
  isLoadingSummary: boolean;
  error?: string;
}

// Custom hook for streaming query
const useStreamQuery = () => {
  const [result, setResult] = useState<StreamingQueryResult>({
    question: '',
    isLoading: false,
    isLoadingSql: false,
    isLoadingData: false,
    isLoadingVisualization: false,
    isLoadingSummary: false
  });
  
  const eventSourceRef = useRef<EventSource | null>(null);
  
  const executeQuery = useCallback((question: string) => {
    // Close previous connection if exists
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    console.log("Executing query:", question);
    
    // Reset result state but preserve the question
    setResult(prev => ({
      // Only reset data fields for a new query, preserving old values until new ones arrive
      sql: undefined,
      data: undefined,
      columns: undefined,
      summary: undefined,
      visualization: undefined,
      followups: undefined,
      error: undefined,
      // Set the new question and loading states
      question,
      isLoading: true,
      isLoadingSql: true,
      isLoadingData: true,
      isLoadingVisualization: true,
      isLoadingSummary: true
    }));
    
    // Create new EventSource connection
    const encodedQuestion = encodeURIComponent(question);
    const eventSource = new EventSource(`https://text2sql.fly.dev/ask_stream?question=${encodedQuestion}`);
    eventSourceRef.current = eventSource;
    
    // Handle different event types
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'question':
          // Question received, no state update needed as we already set it
          break;
        
        case 'sql':
          setResult(prev => ({
            ...prev,
            sql: data.sql,
            isLoadingSql: false
          }));
          break;
        
        case 'data':
          setResult(prev => ({
            ...prev,
            data: data.data,
            columns: data.columns,
            isLoadingData: false
          }));
          break;
        
        case 'visualization':
          setResult(prev => ({
            ...prev,
            visualization: data.visualization,
            isLoadingVisualization: false
          }));
          break;
        
        case 'summary':
          setResult(prev => ({
            ...prev,
            summary: data.summary,
            isLoadingSummary: false
          }));
          break;
          
        case 'followups':
          setResult(prev => ({
            ...prev,
            followups: data.questions
          }));
          break;
        
        case 'error':
          setResult(prev => ({
            ...prev,
            error: data.message,
            isLoading: false,
            isLoadingSql: false,
            isLoadingData: false,
            isLoadingVisualization: false,
            isLoadingSummary: false
          }));
          break;
        
        case 'complete':
          setResult(prev => ({
            ...prev,
            isLoading: false
          }));
          eventSource.close();
          break;
      }
    };
    
    eventSource.onerror = () => {
      setResult(prev => ({
        ...prev,
        error: 'Connection error. Please try again.',
        isLoading: false,
        isLoadingSql: false,
        isLoadingData: false,
        isLoadingVisualization: false,
        isLoadingSummary: false
      }));
      eventSource.close();
    };
    
    return () => {
      eventSource.close();
    };
  }, []);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);
  
  return { result, executeQuery };
};

// Section loading component
const SectionLoading = ({ message }: { message: string }) => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center">
      <Loader className="w-5 h-5 text-primary mx-auto mb-2 animate-spin" />
      <p className="text-xs text-neutral-500 dark:text-neutral-400">{message}</p>
    </div>
  </div>
);

// Custom VolumeWaves component that shows volume with sound waves
const VolumeWaves = ({ className }: { className?: string }) => {
  return (
    <div className={`relative ${className || ''}`}>
      <Volume className="w-full h-full" />
      <div className="absolute left-full top-1/2 -translate-y-1/2 flex items-center space-x-[1px]">
        <div className="w-[2px] h-[3px] bg-current rounded-full animate-pulse" style={{ animationDelay: '0ms' }}></div>
        <div className="w-[2px] h-[5px] bg-current rounded-full animate-pulse" style={{ animationDelay: '150ms' }}></div>
        <div className="w-[2px] h-[4px] bg-current rounded-full animate-pulse" style={{ animationDelay: '300ms' }}></div>
      </div>
    </div>
  );
};

function App() {
  const [selectedDb, setSelectedDb] = useState<string>('');
  const [query, setQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<string | undefined>(undefined);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<'none' | 'liked' | 'disliked'>('none');
  const [currentView, setCurrentView] = useState<'main' | 'history' | 'metrics'>('main');
  const [showSettingsDropdown, setShowSettingsDropdown] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isConnectingDb, setIsConnectingDb] = useState(false);
  const [connectingDbPath, setConnectingDbPath] = useState<string>('');
  const [isProcessingFollowUp, setIsProcessingFollowUp] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  const queryClient = useQueryClient();

  const { data: databases } = useQuery({
    queryKey: ['databases'],
    queryFn: async () => {
      const response = await axios.get('https://text2sql.fly.dev/databases');
      return response.data;
    }
  });

  // Replace askMutation with streaming query
  const { result: streamResult, executeQuery } = useStreamQuery();

  const [loadingState, setLoadingState] = useState<{
    steps: Step[];
    summary?: string;
  }>({
    steps: [
      { id: '1', label: 'Reviewing the query', status: 'pending' },
      { id: '2', label: 'Generating SQL', status: 'pending' },
      { id: '3', label: 'Executing query', status: 'pending' },
      { id: '4', label: 'Processing results', status: 'pending' },
      { id: '5', label: 'Generating visualization', status: 'pending' },
      { id: '6', label: 'Generating data summary', status: 'pending' }
    ],
    summary: undefined
  });

  useEffect(() => {
    // Only initialize the loading sequence when the query first starts loading
    if (streamResult.isLoading) {
      // Reset to initial state
      setLoadingState({
        steps: [
          { id: '1', label: 'Reviewing the query', status: 'loading' },
          { id: '2', label: 'Generating SQL', status: 'pending' },
          { id: '3', label: 'Executing query', status: 'pending' },
          { id: '4', label: 'Processing results', status: 'pending' },
          { id: '5', label: 'Generating visualization', status: 'pending' },
          { id: '6', label: 'Generating data summary', status: 'pending' }
        ],
        summary: undefined
      });
      
      // After 1.5 seconds, complete first step and start second
      const timer1 = setTimeout(() => {
        setLoadingState(prev => ({
          steps: [
            { id: '1', label: 'Reviewing the query', status: 'completed' },
            { id: '2', label: 'Generating SQL', status: 'loading' },
            { id: '3', label: 'Executing query', status: 'pending' },
            { id: '4', label: 'Processing results', status: 'pending' },
            { id: '5', label: 'Generating visualization', status: 'pending' },
            { id: '6', label: 'Generating data summary', status: 'pending' }
          ]
        }));
      }, 1500);
      
      // After another 2 seconds, complete second step and start third
      const timer2 = setTimeout(() => {
        setLoadingState(prev => ({
          steps: [
            { id: '1', label: 'Reviewing the query', status: 'completed' },
            { id: '2', label: 'Generating SQL', status: 'completed' },
            { id: '3', label: 'Executing query', status: 'loading' },
            { id: '4', label: 'Processing results', status: 'pending' },
            { id: '5', label: 'Generating visualization', status: 'pending' },
            { id: '6', label: 'Generating data summary', status: 'pending' }
          ]
        }));
      }, 3500);
      
      // After another 1.5 seconds, complete third step and start fourth
      const timer3 = setTimeout(() => {
        setLoadingState(prev => ({
          steps: [
            { id: '1', label: 'Reviewing the query', status: 'completed' },
            { id: '2', label: 'Generating SQL', status: 'completed' },
            { id: '3', label: 'Executing query', status: 'completed' },
            { id: '4', label: 'Processing results', status: 'loading' },
            { id: '5', label: 'Generating visualization', status: 'pending' },
            { id: '6', label: 'Generating data summary', status: 'pending' }
          ]
        }));
      }, 5000);
      
      // After another 1.5 seconds, complete fourth step and start visualization
      const timer4 = setTimeout(() => {
        setLoadingState(prev => ({
          steps: [
            { id: '1', label: 'Reviewing the query', status: 'completed' },
            { id: '2', label: 'Generating SQL', status: 'completed' },
            { id: '3', label: 'Executing query', status: 'completed' },
            { id: '4', label: 'Processing results', status: 'completed' },
            { id: '5', label: 'Generating visualization', status: 'loading' },
            { id: '6', label: 'Generating data summary', status: 'pending' }
          ]
        }));
      }, 6500);
      
      // After another 1.5 seconds, complete visualization and start summary
      const timer5 = setTimeout(() => {
        setLoadingState(prev => ({
          steps: [
            { id: '1', label: 'Reviewing the query', status: 'completed' },
            { id: '2', label: 'Generating SQL', status: 'completed' },
            { id: '3', label: 'Executing query', status: 'completed' },
            { id: '4', label: 'Processing results', status: 'completed' },
            { id: '5', label: 'Generating visualization', status: 'completed' },
            { id: '6', label: 'Generating data summary', status: 'loading' }
          ]
        }));
      }, 8000);
      
      return () => {
        // Clean up timers if component unmounts or query state changes
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
        clearTimeout(timer4);
        clearTimeout(timer5);
      };
    } else if (streamResult.sql && streamResult.question) {
      // All steps completed when query is successful
      setLoadingState({
        steps: [
          { id: '1', label: 'Reviewing the query', status: 'completed' },
          { id: '2', label: 'Generating SQL', status: 'completed' },
          { id: '3', label: 'Executing query', status: 'completed' },
          { id: '4', label: 'Processing results', status: 'completed' },
          { id: '5', label: 'Generating visualization', status: 'completed' },
          { id: '6', label: 'Generating data summary', status: 'completed' }
        ],
        summary: streamResult.summary || 'Query completed successfully.'
      });
    }
  }, [streamResult.isLoading, streamResult.sql, streamResult.question, streamResult.summary]);

  useEffect(() => {
    // Check if there's a query result in the query cache from history
    const queryCache = queryClient.getQueryCache();
    const queries = queryCache.getAll();
    const historyQuery = queries.find(
      query => query.queryKey[0] === 'history-rerun' && query.state.status === 'success'
    );

    if (historyQuery && historyQuery.state.data) {
      // Type assertion to handle the data correctly
      const data = historyQuery.state.data as { question: string };
      if (data.question) {
        executeQuery(data.question);
      }
      // Remove the temporary query
      queryClient.removeQueries({ queryKey: ['history-rerun'] });
    }
  }, [currentView, queryClient, executeQuery]);

  // Close settings dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setShowSettingsDropdown(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleVoiceQuery = async () => {
    // Don't proceed if no database is selected
    if (!selectedDb) return;

    try {
      setIsProcessingVoice(true);
      // Reset any previously generated audio
      setGeneratedAudio(undefined);
      
      // First, record audio
      const recordResponse = await axios.post('https://text2sql.fly.dev/record_audio', { duration: 5 });
      const audioPath = recordResponse.data.audio_path;
      
      // Then, transcribe the audio
      const transcribeResponse = await axios.post('https://text2sql.fly.dev/transcribe', { 
        audio_path: audioPath
      });
      
      // If transcription was successful
      if (transcribeResponse.data.status === 'success' && transcribeResponse.data.text) {
        const transcribedText = transcribeResponse.data.text;
        
        // Set the query text to what was transcribed
        setQuery(transcribedText);
        
        // Execute streaming query with transcribed text
        executeQuery(transcribedText);
      } else {
        console.error('Transcription failed:', transcribeResponse.data);
      }
    } catch (error) {
      console.error('Error processing voice query:', error);
    } finally {
      setIsRecording(false);
      setIsProcessingVoice(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Don't proceed if no database is selected
    if (!selectedDb) return;
    
    const currentQuery = query.trim();
    if (currentQuery) {
      try {
        // Reset any previously generated audio
        setGeneratedAudio(undefined);
        setFeedbackStatus('none');
        
        // Execute streaming query
        executeQuery(currentQuery);
        
        // Clear the input after submitting
        setQuery('');
      } catch (error) {
        console.error('Error submitting query:', error);
        // Don't clear input on error so user can try again
      }
    }
  };

  const connectToDatabase = async (dbPath: string) => {
    try {
      setIsConnectingDb(true);
      setConnectingDbPath(dbPath);
      await axios.post('https://text2sql.fly.dev/connect', { db_path: dbPath });
      setSelectedDb(dbPath);
    } catch (error) {
      console.error('Error connecting to database:', error);
    } finally {
      setIsConnectingDb(false);
      setConnectingDbPath('');
    }
  };

  const generateSpeech = async (text: string) => {
    if (!text || isGeneratingAudio) return;
    
    try {
      setIsGeneratingAudio(true);
      
      const response = await axios.post('https://text2sql.fly.dev/text_to_speech', {
        text,
        voice: 'Celeste-PlayAI' // Default voice
      });
      
      if (response.data && response.data.audio_base64) {
        setGeneratedAudio(response.data.audio_base64);
      }
    } catch (error) {
      console.error('Error generating speech:', error);
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const handleFeedback = async (type: 'liked' | 'disliked') => {
    setFeedbackStatus(type);
    
    // Only proceed if we have query results
    if (streamResult.sql && streamResult.question) {
      try {
        // Map our UI feedback type to the API's expected format
        const feedbackValue = type === 'liked' ? 'up' : 'down';
        
        // Send feedback to the backend
        const response = await axios.post('https://text2sql.fly.dev/feedback', {
          feedback: feedbackValue,
          question: streamResult.question,
          sql: streamResult.sql
        });
        
        console.log(`Feedback (${feedbackValue}) sent successfully`);
        
        // Check if this was a duplicate entry
        if (response.data.duplicate) {
          console.log('This example was already in the training data');
        }
      } catch (error) {
        console.error('Error sending feedback:', error);
      }
    }
    
    // Reset feedback status after a delay
    setTimeout(() => {
      setFeedbackStatus('none');
    }, 3000);
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

  const handleCleanupDuplicates = async () => {
    try {
      const response = await axios.post('https://text2sql.fly.dev/cleanup_duplicates');
      if (response.data.status === 'success') {
        alert(`Cleaned up ${response.data.duplicates_removed} duplicates from ${response.data.files_cleaned} files.`);
      } else {
        alert('Error cleaning up duplicates.');
      }
    } catch (error) {
      console.error('Error cleaning up duplicates:', error);
      alert('Error cleaning up duplicates.');
    }
    setShowSettingsDropdown(false);
  };

  const uploadDatabase = async (file: File) => {
    if (!file) return;

    try {
      setIsUploading(true);
      
      // Check if file is a SQLite file
      if (!file.name.endsWith('.sqlite') && !file.name.endsWith('.db')) {
        alert('Only .sqlite or .db files are allowed');
        setIsUploading(false);
        return;
      }
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Upload the database
      const response = await axios.post('https://text2sql.fly.dev/upload_database', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      if (response.data.status === 'success') {
        // Refresh the database list
        queryClient.invalidateQueries({ queryKey: ['databases'] });
        alert(`Database ${file.name} uploaded successfully`);
        
        // Auto-connect to the uploaded database
        if (response.data.path) {
          connectToDatabase(response.data.path);
        }
      } else {
        alert(`Error: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error uploading database:', error);
      alert('Failed to upload database. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadDatabase(e.target.files[0]);
    }
  };

  // Render history page if that's the current view
  if (currentView === 'history') {
    return (
      <HistoryPage 
        onBack={() => setCurrentView('main')} 
        queryClient={queryClient}
      />
    );
  }
  
  // Render metrics page if that's the current view
  if (currentView === 'metrics') {
    return (
      <MetricsPage
        onBack={() => setCurrentView('main')}
      />
    );
  }

  return (
    <div className="min-h-screen bg-neutral-100 dark:bg-neutral-900 flex">
      {/* Sidebar */}
      <aside className="w-[280px] bg-white dark:bg-neutral-800 border-r border-neutral-200 dark:border-neutral-700 flex flex-col">
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center gap-2 mb-4">
            <Database className="w-6 h-6 text-secondary" />
            <h1 className="text-lg font-semibold text-neutral-900 dark:text-white">Text2SQL</h1>
          </div>
          <Tooltip content="Upload a new database file">
            <label className="w-full flex items-center gap-2 px-4 py-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg text-neutral-600 dark:text-neutral-200 hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors cursor-pointer">
              {isUploading ? (
                <div className="animate-spin w-4 h-4 border-2 border-primary border-t-transparent rounded-full" />
              ) : (
                <Upload className="w-4 h-4" />
              )}
              <span>{isUploading ? 'Uploading...' : 'Upload Database'}</span>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept=".sqlite,.db" 
                className="hidden" 
                disabled={isUploading}
              />
            </label>
          </Tooltip>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2">Databases</h2>
          <div className="space-y-1">
            {databases?.databases.map((db: Database) => (
              <Tooltip key={db.path} content={`Connect to ${db.name}`}>
                <button
                  onClick={() => connectToDatabase(db.path)}
                  className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    selectedDb === db.path
                      ? 'bg-primary text-white'
                      : 'text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                  }`}
                  disabled={isConnectingDb && db.path === connectingDbPath}
                >
                  {isConnectingDb && db.path === connectingDbPath ? (
                    <Loader className="w-4 h-4 animate-spin" />
                  ) : (
                    <Database className="w-4 h-4" />
                  )}
                  <span className="truncate">{db.name}</span>
                  {isConnectingDb && db.path === connectingDbPath && (
                    <span className="ml-auto text-xs text-primary">Connecting...</span>
                  )}
                </button>
              </Tooltip>
            ))}
          </div>
        </div>

        <nav className="p-4 border-t border-neutral-200 dark:border-neutral-700">
          <div className="space-y-1">
            <Tooltip content="View query history">
              <button 
                onClick={() => setCurrentView('history')}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
              >
                <History className="w-4 h-4" />
                <span>History</span>
              </button>
            </Tooltip>
            <Tooltip content="View analytics metrics">
              <button 
                onClick={() => setCurrentView('metrics')}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
              >
                <Activity className="w-4 h-4" />
                <span>Metrics</span>
              </button>
            </Tooltip>
            <div className="relative" ref={settingsRef}>
              <Tooltip content="Adjust settings">
                <button 
                  onClick={() => setShowSettingsDropdown(!showSettingsDropdown)}
                  className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Settings className="w-4 h-4" />
                    <span>Settings</span>
                  </div>
                  <div className="text-xs">{showSettingsDropdown ? '▲' : '▼'}</div>
                </button>
              </Tooltip>
              {showSettingsDropdown && (
                <div className="absolute z-10 mt-1 w-full bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg shadow-lg overflow-hidden">
                  <button 
                    onClick={handleCleanupDuplicates}
                    className="w-full flex items-center gap-2 px-3 py-2 text-left text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                  >
                    <FileText className="w-4 h-4" />
                    <span>Clean Up Duplicates</span>
                  </button>
                  {/* Add more settings options here */}
                </div>
              )}
            </div>
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-sm text-neutral-600 dark:text-neutral-300">Theme</span>
              <ThemeToggle />
            </div>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen">
        {/* Results Area */}
        <div className="flex-1 p-3 overflow-hidden">
          <div className="h-full flex flex-col">
            {streamResult.error && (
              <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-4 border-l-4 border-red-500 mb-3">
                <h3 className="text-red-500 font-medium mb-2">Error</h3>
                <p className="text-neutral-700 dark:text-neutral-300">
                  {streamResult.error}
                </p>
              </div>
            )}
            
            {(streamResult.isLoading || streamResult.question) && !streamResult.error && (
              <>
                {/* User Question Display */}
                <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card p-2 mb-3">
                  <h3 className="text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">Your Question</h3>
                  <p className="text-sm text-neutral-800 dark:text-neutral-200">{streamResult.question}</p>
                </div>

                {/* 2x2 Grid Layout */}
                <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-3 h-[calc(100vh-160px)]">
                  {/* Top Left: SQL Query Section */}
                  <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden flex flex-col">
                    <div className="px-3 py-1.5 flex items-center border-b border-neutral-200 dark:border-neutral-700">
                      <FileCode className="w-3.5 h-3.5 text-secondary mr-2" />
                      <h3 className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Generated SQL</h3>
                    </div>
                    <div className="flex-1 p-2 overflow-auto">
                      {streamResult.isLoadingSql ? (
                        <SectionLoading message="Generating SQL query..." />
                      ) : streamResult.sql ? (
                        <pre className="bg-neutral-100 dark:bg-neutral-700 rounded-lg p-2 text-xs font-mono overflow-auto h-full">
                          {streamResult.sql}
                        </pre>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-xs text-neutral-500 dark:text-neutral-400">No SQL query yet</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Top Right: Results Section */}
                  <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden flex flex-col">
                    <div className="px-3 py-1.5 flex items-center justify-between border-b border-neutral-200 dark:border-neutral-700">
                      <div className="flex items-center">
                        <Table className="w-3.5 h-3.5 text-secondary mr-2" />
                        <h3 className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Results</h3>
                      </div>
                      {!streamResult.isLoadingData && streamResult.data && streamResult.columns && (
                        <Tooltip content="Download as CSV file" position="bottom" offset={24}>
                          <button
                            onClick={() => handleDownloadCSV(streamResult.data!, streamResult.columns!)}
                            className="flex items-center gap-1 px-1.5 py-0.5 text-xs bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200 rounded-md hover:bg-neutral-200 dark:hover:bg-neutral-600 transition-colors"
                          >
                            <Download className="w-3 h-3" />
                            <span>CSV</span>
                          </button>
                        </Tooltip>
                      )}
                    </div>
                    <div className="flex-1 p-1 overflow-auto">
                      {streamResult.isLoadingData ? (
                        <SectionLoading message="Executing query..." />
                      ) : streamResult.data && streamResult.columns && streamResult.data.length > 0 ? (
                        <table className="min-w-full text-xs">
                          <thead>
                            <tr className="bg-neutral-50 dark:bg-neutral-750">
                              {streamResult.columns.map((column: string, idx: number) => (
                                <th 
                                  key={idx} 
                                  className="px-2 py-1 text-left font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider sticky top-0 bg-neutral-50 dark:bg-neutral-750"
                                >
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-neutral-200 dark:divide-neutral-700">
                            {streamResult.data.map((row: any, rowIdx: number) => (
                              <tr key={rowIdx} className="hover:bg-neutral-100 dark:hover:bg-neutral-700">
                                {streamResult.columns!.map((column: string, colIdx: number) => (
                                  <td 
                                    key={colIdx} 
                                    className="px-2 py-1 text-neutral-900 dark:text-neutral-100 whitespace-nowrap"
                                  >
                                    {row[column]}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-xs text-neutral-500 dark:text-neutral-400">No results to display</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Bottom Left: Summary Section */}
                  <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden flex flex-col">
                    <div className="px-3 py-1.5 flex items-center justify-between border-b border-neutral-200 dark:border-neutral-700">
                      <div className="flex items-center">
                        <FileText className="w-3.5 h-3.5 text-secondary mr-2" />
                        <h3 className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Summary</h3>
                      </div>
                      {!streamResult.isLoadingSummary && streamResult.summary && (
                        <div className="flex items-center gap-1">
                          <Tooltip content="This was helpful" position="bottom" offset={24}>
                            <button
                              onClick={() => handleFeedback('liked')}
                              disabled={feedbackStatus !== 'none'}
                              className={`p-1 rounded-md transition-colors ${
                                feedbackStatus === 'liked' 
                                  ? 'text-green-500 bg-green-100 dark:bg-green-900 dark:bg-opacity-30' 
                                  : 'text-neutral-500 hover:text-green-500 hover:bg-green-100 dark:hover:bg-neutral-600'
                              }`}
                            >
                              <ThumbsUp className="w-3 h-3" />
                            </button>
                          </Tooltip>
                          <Tooltip content="This was not helpful" position="bottom" offset={24}>
                            <button
                              onClick={() => handleFeedback('disliked')}
                              disabled={feedbackStatus !== 'none'}
                              className={`p-1 rounded-md transition-colors ${
                                feedbackStatus === 'disliked' 
                                  ? 'text-red-500 bg-red-100 dark:bg-red-900 dark:bg-opacity-30' 
                                  : 'text-neutral-500 hover:text-red-500 hover:bg-red-100 dark:hover:bg-neutral-600'
                              }`}
                            >
                              <ThumbsDown className="w-3 h-3" />
                            </button>
                          </Tooltip>
                          <Tooltip content="Speak" position="bottom" offset={24}>
                            <button
                              onClick={() => generateSpeech(streamResult.summary || '')}
                              disabled={isGeneratingAudio || !streamResult.summary}
                              className={`p-1 rounded-md bg-primary bg-opacity-10 text-primary hover:bg-opacity-20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGeneratingAudio ? 'animate-pulse' : ''}`}
                            >
                              <VolumeWaves className="w-3 h-3" />
                            </button>
                          </Tooltip>
                        </div>
                      )}
                    </div>
                    <div className="flex-1 p-2 overflow-auto">
                      {streamResult.isLoadingSummary ? (
                        <SectionLoading message="Generating data summary..." />
                      ) : streamResult.summary ? (
                        <div className="text-xs text-neutral-800 dark:text-neutral-200 h-full overflow-auto">
                          {streamResult.summary}
                          
                          {/* Follow-up questions */}
                          {streamResult.followups && streamResult.followups.length > 0 && (
                            <div className="mt-4 pt-2 border-t border-neutral-200 dark:border-neutral-700">
                              <h4 className="text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Follow-up Questions</h4>
                              <FollowUpQuestions 
                                questions={streamResult.followups} 
                                isProcessing={isProcessingFollowUp}
                                onSelect={(q) => {
                                  // Prevent multiple rapid clicks
                                  if (isProcessingFollowUp) return;
                                  
                                  console.log("Follow-up question selected:", q);
                                  setIsProcessingFollowUp(true);
                                  
                                  // Reset any previously generated audio
                                  setGeneratedAudio(undefined);
                                  setFeedbackStatus('none');
                                  
                                  // Execute the follow-up query
                                  executeQuery(q);
                                  
                                  // Reset the processing state after a short delay
                                  setTimeout(() => {
                                    setIsProcessingFollowUp(false);
                                  }, 1000);
                                }}
                              />
                            </div>
                          )}
                          
                          {/* Audio Player for Voice Response or Generated Audio */}
                          {generatedAudio && (
                            <div className="mt-2 pt-2 border-t border-neutral-200 dark:border-neutral-700">
                              <AudioPlayer audioBase64={generatedAudio} />
                            </div>
                          )}
                          
                          {/* Feedback status */}
                          {feedbackStatus === 'liked' && (
                            <div className="mt-1 text-xs text-green-500">Added to memory</div>
                          )}
                          {feedbackStatus === 'disliked' && (
                            <div className="mt-1 text-xs text-red-500">Thanks for the feedback</div>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-xs text-neutral-500 dark:text-neutral-400">No summary yet</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Bottom Right: Visualization Section */}
                  <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden flex flex-col">
                    <div className="px-3 py-1.5 flex items-center border-b border-neutral-200 dark:border-neutral-700">
                      <BarChart className="w-3.5 h-3.5 text-secondary mr-2" />
                      <h3 className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Visualization</h3>
                    </div>
                    <div className="flex-1 p-1 overflow-hidden">
                      {streamResult.isLoadingVisualization ? (
                        <SectionLoading message="Creating visualization..." />
                      ) : streamResult.visualization ? (
                        <div className="h-full w-full">
                          <PlotlyChart visualizationJson={streamResult.visualization} />
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <div className="text-center p-4">
                            <BarChart className="w-6 h-6 text-neutral-300 dark:text-neutral-600 mx-auto mb-1" />
                            <p className="text-xs text-neutral-500 dark:text-neutral-400">No visualization available</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Query Input */}
        <div className="border-t border-neutral-200 dark:border-neutral-700 p-2">
          <form onSubmit={handleSubmit} className="max-w-6xl mx-auto" key="query-form">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={(e) => e.target.select()}
                placeholder={
                  isConnectingDb 
                    ? "Connecting to database..."
                    : selectedDb 
                      ? "Ask a question about your data..." 
                      : "Select a database first"
                }
                className="w-full bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 border border-neutral-300 dark:border-neutral-600 rounded-lg pl-4 pr-20 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                disabled={isProcessingVoice || !selectedDb || streamResult.isLoading || isConnectingDb}
                autoComplete="off"
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <Tooltip content={selectedDb ? "Record voice question" : "Select a database first"} offset={8}>
                  <button
                    type="button"
                    onClick={() => {
                      if (!isProcessingVoice && selectedDb && !streamResult.isLoading) {
                        setIsRecording(!isRecording);
                        if (!isRecording) {
                          handleVoiceQuery();
                        }
                      }
                    }}
                    className={`p-1.5 rounded-lg transition-colors ${
                      !selectedDb || streamResult.isLoading
                        ? 'text-neutral-400 bg-neutral-100 dark:bg-neutral-700 cursor-not-allowed'
                        : isProcessingVoice 
                          ? 'text-primary bg-primary bg-opacity-10 animate-pulse' 
                          : isRecording
                            ? 'text-primary bg-primary bg-opacity-10'
                            : 'text-neutral-500 hover:text-primary hover:bg-primary hover:bg-opacity-10 dark:hover:bg-neutral-600'
                    }`}
                    disabled={isProcessingVoice || !selectedDb || streamResult.isLoading}
                  >
                    <Mic className="w-4 h-4" />
                  </button>
                </Tooltip>
                <Tooltip content={selectedDb ? "Submit question" : "Select a database first"} offset={8}>
                  <button
                    type="submit"
                    disabled={!query.trim() || streamResult.isLoading || isProcessingVoice || !selectedDb}
                    className="p-1.5 rounded-lg text-neutral-500 hover:text-primary hover:bg-primary hover:bg-opacity-10 dark:hover:bg-neutral-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </Tooltip>
              </div>
            </div>
            {!selectedDb && !isConnectingDb && (
              <div className="mt-2 text-xs text-center text-amber-600 dark:text-amber-400">
                Please select a database from the sidebar before asking a question.
              </div>
            )}
            {isConnectingDb && (
              <div className="mt-2 text-xs text-center text-primary flex items-center justify-center">
                <Loader className="w-3 h-3 animate-spin mr-1" />
                Connecting to database... This may take a moment.
              </div>
            )}
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;