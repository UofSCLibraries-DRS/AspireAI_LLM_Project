// Types for functionally across the app

/**
 * Model indetifier type
 * - name: display name for model
 * - apiValue: backend model ID (e.g., "M8")
 */
export interface ModelOption {
  name: string;
  apiValue: string;
}

/**
 * GAICo request payload:
 * - modelName: display name for model
 * - apiValue: backend model ID (e.g., "M8")
 * - prompt: user input text
 * - chatbotResponse: model response text
 */
export interface GaicoRequest {
  modelName: string;
  apiValue: string;
  prompt: string;
  chatbotResponse: string;
}