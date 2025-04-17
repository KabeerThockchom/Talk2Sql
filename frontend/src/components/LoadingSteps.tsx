import React from 'react';
import { Check } from 'lucide-react';

export interface Step {
  id: string;
  label: string;
  status: 'completed' | 'loading' | 'pending';
}

interface LoadingStepsProps {
  steps: Step[];
  title?: string;
}

export function LoadingSteps({ steps, title }: LoadingStepsProps) {
  const completedSteps = steps.filter(step => step.status === 'completed').length;
  const totalSteps = steps.length;
  const isFinished = completedSteps === totalSteps;
  
  return (
    <div className="flex flex-col gap-6 p-4">
      <div className="flex items-center">
        {/* Working/Finished text */}
        <div className="text-xl text-neutral-500 dark:text-neutral-400 font-normal">
          {isFinished ? `Finished in ${totalSteps} steps` : 'Working...'}
        </div>
      </div>
      
      <div className="pl-2 flex flex-col gap-8">
        {steps.map((step) => (
          <div key={step.id} className="flex items-center gap-5">
            <div className="flex h-6 w-6 items-center justify-center">
              {step.status === 'completed' && (
                <Check className="h-5 w-5 text-neutral-500 dark:text-neutral-400" strokeWidth={2.5} />
              )}
              {step.status === 'loading' && (
                <svg className="animate-spin h-5 w-5 text-secondary" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              )}
              {step.status === 'pending' && (
                <div className="h-5 w-5 rounded-full border-2 border-neutral-300 dark:border-neutral-500" />
              )}
            </div>
            <span className="text-lg text-neutral-500 dark:text-neutral-400 font-normal">
              {step.label}
            </span>
          </div>
        ))}
      </div>
      
      {isFinished && title && (
        <div className="mt-2">
          <div className="text-neutral-700 dark:text-neutral-300">
            {title}
          </div>
        </div>
      )}
    </div>
  );
}