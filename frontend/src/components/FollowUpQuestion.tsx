import React from 'react';
import { MessageSquare, Loader } from 'lucide-react';
import { Tooltip } from './Tooltip';

interface FollowUpQuestionProps {
  question: string;
  onSelect: (question: string) => void;
  isProcessing?: boolean;
}

export function FollowUpQuestion({ question, onSelect, isProcessing = false }: FollowUpQuestionProps) {
  return (
    <Tooltip content="Click to ask this question">
      <button
        onClick={() => onSelect(question)}
        disabled={isProcessing}
        className={`py-1 px-2 text-xs bg-primary/10 text-primary hover:bg-primary/20 
                  rounded-full flex items-center gap-1 transition-colors max-w-full
                  hover:scale-105 hover:shadow-sm cursor-pointer
                  ${isProcessing ? 'opacity-70 cursor-not-allowed' : ''}`}
      >
        {isProcessing ? (
          <Loader className="w-3 h-3 shrink-0 animate-spin" />
        ) : (
          <MessageSquare className="w-3 h-3 shrink-0" />
        )}
        <span className="truncate">{question}</span>
      </button>
    </Tooltip>
  );
}

interface FollowUpQuestionsProps {
  questions?: string[];
  onSelect: (question: string) => void;
  isProcessing?: boolean;
}

export function FollowUpQuestions({ questions, onSelect, isProcessing = false }: FollowUpQuestionsProps) {
  if (!questions || questions.length === 0) return null;
  
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {questions.map((question, idx) => (
        <FollowUpQuestion 
          key={idx} 
          question={question} 
          onSelect={onSelect} 
          isProcessing={isProcessing}
        />
      ))}
    </div>
  );
} 