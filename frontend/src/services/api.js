/**
 * API Service Layer
 *
 * Centralized API communication with the backend.
 * Handles all HTTP requests to the FastAPI backend.
 */

import { BASE_URL } from '../config';

/**
 * Execute unified workflow - automatic routing based on input
 * @param {Object} options
 * @param {File|null} options.file - PDF file to upload (optional)
 * @param {string|null} options.question - Question to ask (optional)
 * @param {number} options.maxLength - Chunk size for PDF processing (default: 1000)
 * @param {number} options.overlap - Chunk overlap for PDF processing (default: 100)
 * @returns {Promise<Object>} Unified response from backend
 */
export async function executeUnified({ file, question, maxLength = 1000, overlap = 100 }) {
  const formData = new FormData();

  // Add file if present
  if (file) {
    formData.append('file', file);
  }

  // Add question if present
  if (question && question.trim()) {
    formData.append('question', question.trim());
  }

  // Add processing parameters
  formData.append('max_length', maxLength.toString());
  formData.append('overlap', overlap.toString());

  const response = await fetch(`${BASE_URL}/execute/`, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - browser will set it with boundary for FormData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Query the database with a question (legacy endpoint)
 * @param {string} question - The question to ask
 * @returns {Promise<Object>} Query response
 */
export async function askQuestion(question) {
  const response = await fetch(`${BASE_URL}/query/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Upload and process a PDF file (legacy endpoint)
 * @param {Object} options
 * @param {File} options.file - PDF file to upload
 * @param {number} options.maxLength - Chunk size (default: 1000)
 * @param {number} options.overlap - Chunk overlap (default: 100)
 * @returns {Promise<Object>} Upload response
 */
export async function uploadPdf({ file, maxLength = 1000, overlap = 100 }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('max_length', maxLength.toString());
  formData.append('overlap', overlap.toString());

  const response = await fetch(`${BASE_URL}/process-pdf/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Check backend health
 * @returns {Promise<Object>} Health check response
 */
export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}