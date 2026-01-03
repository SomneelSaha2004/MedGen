import AnalyticsIcon from '@mui/icons-material/Analytics';
import DatasetIcon from '@mui/icons-material/Dataset';
import DownloadIcon from '@mui/icons-material/Download';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SaveIcon from '@mui/icons-material/Save';
import SpeedIcon from '@mui/icons-material/Speed';
import StorageIcon from '@mui/icons-material/Storage';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  ButtonGroup,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  LinearProgress,
  Paper,
  Slider,
  Snackbar,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography
} from '@mui/material';
import { motion } from 'framer-motion';
import { useContext, useEffect, useState } from 'react';
import api from '../services/api';
// Import context from App.js
import { CSVContext, GenerationContext, UploadResponseContext } from '../App';

const DataGeneration = () => {
  // State for generation parameters
  const [numSamples, setNumSamples] = useState(() => {
    const saved = localStorage.getItem('data_generation_numSamples');
    return saved ? parseInt(saved, 10) : 10;
  });
  const [temperature, setTemperature] = useState(() => {
    const saved = localStorage.getItem('data_generation_temperature');
    return saved ? parseFloat(saved) : 0.7;
  });
  const [topP, setTopP] = useState(() => {
    const saved = localStorage.getItem('data_generation_topP');
    return saved ? parseFloat(saved) : 0.9;
  });
  const [repetitionPenalty, setRepetitionPenalty] = useState(() => {
    const saved = localStorage.getItem('data_generation_repetitionPenalty');
    return saved ? parseFloat(saved) : 1.1;
  });
  const [maxTokens, setMaxTokens] = useState(() => {
    const saved = localStorage.getItem('data_generation_maxTokens');
    return saved ? parseInt(saved, 10) : 2048;
  });
  // Fixed model value
  const model = 'gpt-4o-mini';

  // Generation mode: 'fast' or 'deep'
  const [generationMode, setGenerationMode] = useState(() => {
    const saved = localStorage.getItem('data_generation_mode');
    return saved || 'fast';
  });

  // State for data generation and display
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generatedData, setGeneratedData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [error, setError] = useState(null);
  const [totalRows, setTotalRows] = useState(0);
  const [originalCount, setOriginalCount] = useState(0);
  const [syntheticCount, setSyntheticCount] = useState(0);
  const [isCombined, setIsCombined] = useState(false);
  const [showCombined, setShowCombined] = useState(true);

  // State for sample datasets
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [loadingSampleDatasets, setLoadingSampleDatasets] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [hasCSV, setHasCSV] = useState(false);
  const [currentDatasetInfo, setCurrentDatasetInfo] = useState(null);

  // State for download and actions
  const [downloading, setDownloading] = useState(false);
  const [switchingData, setSwitchingData] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // State for save dataset dialog
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveDescription, setSaveDescription] = useState('');
  const [saveType, setSaveType] = useState('synthetic');
  const [saving, setSaving] = useState(false);

  // Access the contexts
  const { setCSVUploaded } = useContext(CSVContext);
  const { setGenerationStatus } = useContext(GenerationContext);
  const { setUploadResponse } = useContext(UploadResponseContext);


  // Check for ongoing generation on component mount
  useEffect(() => {
    // Check if there's an ongoing generation when component mounts
    const storedGenerationStatus = localStorage.getItem('generationStatus');
    if (storedGenerationStatus) {
      try {
        const status = JSON.parse(storedGenerationStatus);
        // Only restore if generation is still in progress
        if (status.isGenerating) {
          setIsGenerating(true);
          setGenerationProgress(status.progress || 0);
          // Update global state
          setGenerationStatus({
            isGenerating: true,
            progress: status.progress || 0,
            currentFile: null,
            error: null
          });
          checkGenerationStatus(); // Immediately check current status
        } else {
          // Clear localStorage if generation is complete
          localStorage.removeItem('generationStatus');
        }
      } catch (e) {
        console.error("Error parsing stored generation status", e);
        localStorage.removeItem('generationStatus');
      }
    }

    // Check for previously generated data
    try {
      const savedData = localStorage.getItem('generatedData');
      const savedColumns = localStorage.getItem('generatedColumns');
      const savedMetadata = localStorage.getItem('generatedMetadata');

      if (savedData && savedColumns && savedMetadata) {
        const parsedData = JSON.parse(savedData);
        const parsedColumns = JSON.parse(savedColumns);
        const parsedMetadata = JSON.parse(savedMetadata);

        setGeneratedData(parsedData);
        setColumns(parsedColumns);
        setTotalRows(parsedMetadata.totalRows || parsedData.length);
        setOriginalCount(parsedMetadata.originalCount || 0);
        setSyntheticCount(parsedMetadata.syntheticCount || 0);
        setIsCombined(parsedMetadata.isCombined || false);
        console.log("Restored previously generated data from localStorage");
      }
    } catch (e) {
      console.error("Error restoring generated data:", e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setGenerationStatus]);

  // Check CSV status and fetch sample datasets on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await api.checkCSVStatus();
        setHasCSV(status.hasCSV || false);
        if (status.hasCSV) {
          setCurrentDatasetInfo({
            filename: status.filename,
            rows: status.rows,
            columns: status.columnNames || status.columns
          });
        }
      } catch (error) {
        console.error('Error checking CSV status:', error);
        setHasCSV(false);
      }
    };

    const fetchSampleDatasets = async () => {
      setLoadingSampleDatasets(true);
      try {
        const response = await api.getSampleDatasets();
        if (response.success) {
          setSampleDatasets(response.datasets);
        }
      } catch (error) {
        console.error('Error fetching sample datasets:', error);
      } finally {
        setLoadingSampleDatasets(false);
      }
    };

    checkStatus();
    fetchSampleDatasets();
  }, []);

  // Handle selecting a sample dataset
  const handleUseSampleDataset = async (filename) => {
    setLoadingDataset(true);
    setError(null);
    setSelectedDataset(filename);

    try {
      const response = await api.useSampleDataset(filename);
      if (response.success) {
        setHasCSV(true);
        setCurrentDatasetInfo({
          filename: response.filename,
          name: response.name,
          rows: response.rows,
          columns: response.columns
        });
        setCSVUploaded(true);
        setUploadResponse({
          success: true,
          message: response.message,
          rows: response.rows,
          columns: response.columns,
          filename: response.filename,
          rowCount: response.rows,
          columnCount: response.columnCount,
          has_csv: true
        });
      } else {
        setError(response.error || 'Failed to load dataset');
      }
    } catch (error) {
      console.error('Error using sample dataset:', error);
      setError(error.message || 'Failed to load dataset');
    } finally {
      setLoadingDataset(false);
      setSelectedDataset(null);
    }
  };

  // Save parameters to localStorage when they change
  useEffect(() => {
    localStorage.setItem('data_generation_numSamples', numSamples.toString());
  }, [numSamples]);

  useEffect(() => {
    localStorage.setItem('data_generation_temperature', temperature.toString());
  }, [temperature]);

  useEffect(() => {
    localStorage.setItem('data_generation_topP', topP.toString());
  }, [topP]);

  useEffect(() => {
    localStorage.setItem('data_generation_repetitionPenalty', repetitionPenalty.toString());
  }, [repetitionPenalty]);

  useEffect(() => {
    localStorage.setItem('data_generation_maxTokens', maxTokens.toString());
  }, [maxTokens]);

  useEffect(() => {
    localStorage.setItem('data_generation_mode', generationMode);
  }, [generationMode]);

  // Effect to fetch data when showCombined changes
  useEffect(() => {
    if (generatedData) {
      fetchGeneratedData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showCombined]);

  // Poll for generation status when isGenerating is true
  useEffect(() => {
    let interval;

    if (isGenerating) {
      interval = setInterval(checkGenerationStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGenerating]);

  // Update global state when local state changes
  useEffect(() => {
    setGenerationStatus({
      isGenerating,
      progress: generationProgress,
      currentFile: null,
      error: error
    });
  }, [isGenerating, generationProgress, error, setGenerationStatus]);

  // Check generation status
  const checkGenerationStatus = async () => {
    try {
      const response = await api.getGenerationStatus();

      // Only update if progress has changed
      if (response.progress !== generationProgress) {
        setGenerationProgress(response.progress || 0);

        // Store generation status in localStorage for persistence across tab switches
        localStorage.setItem('generationStatus', JSON.stringify({
          isGenerating: response.isGenerating,
          progress: response.progress || 0,
          timestamp: new Date().getTime()
        }));

        // Log progress to console for debugging
        console.log(`Generation progress update: ${response.progress.toFixed(1)}%`);
      }

      // If generation is complete, fetch the results
      if (!response.isGenerating && response.progress >= 100) {
        setGenerationProgress(100);
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage

        // Only fetch data if we have generated data and no error
        if (response.has_generated_data) {
          console.log("Generation complete, fetching data...");
          fetchGeneratedData();
        } else if (response.error) {
          setError(response.error_details || response.error);
        }
      } else if (!response.isGenerating && response.progress < 100) {
        // Handle case where generation stopped before completion
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage
        if (response.error) {
          setError(response.error_details || response.error);
        } else if (response.has_generated_data) {
          console.log("Generation stopped but data available, fetching data...");
          fetchGeneratedData();
        } else {
          setError('Data generation was interrupted before completion');
        }
      }
    } catch (error) {
      console.error("Error checking generation status:", error);
      setError('Failed to check generation status: ' + (error.message || 'Unknown error'));
      setIsGenerating(false);
      localStorage.removeItem('generationStatus'); // Clear localStorage on error
    }
  };

  // Fetch generated data
  const fetchGeneratedData = async () => {
    try {
      const response = await api.get(`/get_generated_data${showCombined ? '' : '?combined=false'}`);

      if (response.data.success) {
        // Save the data to state
        setGeneratedData(response.data.data);
        setColumns(response.data.columns);
        setTotalRows(response.data.rowCount || response.data.data.length);
        setError(null);
        setOriginalCount(response.data.originalCount || 0);
        setSyntheticCount(response.data.syntheticCount || 0);
        setIsCombined(response.data.isCombined || false);

        // Store the data in localStorage for persistence
        try {
          localStorage.setItem('generatedData', JSON.stringify(response.data.data));
          localStorage.setItem('generatedColumns', JSON.stringify(response.data.columns));
          localStorage.setItem('generatedMetadata', JSON.stringify({
            totalRows: response.data.rowCount || response.data.data.length,
            originalCount: response.data.originalCount || 0,
            syntheticCount: response.data.syntheticCount || 0,
            isCombined: response.data.isCombined || false
          }));
        } catch (storageError) {
          // If storing fails (e.g., due to size limits), just log the error
          console.error("Error storing generated data in localStorage:", storageError);
        }

        // Update the CSV status globally after successful generation
        try {
          const csvStatus = await api.checkCSVStatus();
          if (csvStatus && csvStatus.has_csv) {
            // Update the upload response using the context
            setUploadResponse(csvStatus);

            // Ensure CSV uploaded flag is set
            setCSVUploaded(true);
          }
        } catch (statusError) {
          console.error("Error updating CSV status after generation:", statusError);
        }
      } else {
        setError(response.data.error || 'Failed to fetch generated data');
      }
    } catch (error) {
      console.error("Error fetching generated data:", error);
      setError(error.message || 'Failed to fetch generated data');
    }
  };

  // Handle form submission
  const handleGenerate = async () => {
    setError(null);
    setIsGenerating(true);
    setGenerationProgress(0);

    // Clear any previously stored data since we're generating new data
    setGeneratedData(null);
    localStorage.removeItem('generatedData');
    localStorage.removeItem('generatedColumns');
    localStorage.removeItem('generatedMetadata');

    // Store initial generation status in localStorage
    localStorage.setItem('generationStatus', JSON.stringify({
      isGenerating: true,
      progress: 0,
      timestamp: new Date().getTime()
    }));

    try {
      // Send parameters to the backend
      const response = await api.post('/generate_data', {
        numSamples,
        temperature,
        topP,
        repetitionPenalty,
        maxTokens,
        model,
        generationMode
      });

      if (!response.data.success) {
        setError(response.data.error || 'Failed to start data generation');
        setIsGenerating(false);
        localStorage.removeItem('generationStatus'); // Clear localStorage on error
      }
    } catch (error) {
      console.error("Error starting data generation:", error);
      setError(error.message || 'Failed to start data generation');
      setIsGenerating(false);
      localStorage.removeItem('generationStatus'); // Clear localStorage on error
    }
  };

  // Handle download of generated data
  const handleDownload = async (type) => {
    setDownloading(true);
    try {
      await api.downloadData(type);
      setSnackbar({
        open: true,
        message: `${type.charAt(0).toUpperCase() + type.slice(1)} data downloaded successfully!`,
        severity: 'success'
      });
    } catch (error) {
      console.error("Error downloading data:", error);
      setSnackbar({
        open: true,
        message: error.message || 'Failed to download data',
        severity: 'error'
      });
    } finally {
      setDownloading(false);
    }
  };

  // Handle switching to generated data for analysis
  const handleUseForAnalysis = async (useCombined = true) => {
    setSwitchingData(true);
    try {
      const response = await api.useGeneratedData(useCombined);
      if (response.success) {
        // Update global state
        setCSVUploaded(true);
        setUploadResponse(response);

        setSnackbar({
          open: true,
          message: `Now using ${useCombined ? 'combined' : 'synthetic'} data for analysis. Go to Analysis page to explore!`,
          severity: 'success'
        });
      } else {
        throw new Error(response.error || 'Failed to switch data');
      }
    } catch (error) {
      console.error("Error switching to generated data:", error);
      setSnackbar({
        open: true,
        message: error.message || 'Failed to switch to generated data',
        severity: 'error'
      });
    } finally {
      setSwitchingData(false);
    }
  };

  // Handle saving generated data as a new dataset
  const handleSaveDataset = async () => {
    if (!saveName.trim()) {
      setSnackbar({
        open: true,
        message: 'Please enter a dataset name',
        severity: 'warning'
      });
      return;
    }

    setSaving(true);
    try {
      const response = await api.saveDataset(saveName, saveDescription, saveType);
      if (response.success) {
        setSnackbar({
          open: true,
          message: `Dataset saved: ${saveName}. Go to Datasets page to manage.`,
          severity: 'success'
        });
        setSaveDialogOpen(false);
        setSaveName('');
        setSaveDescription('');
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error("Error saving dataset:", error);
      setSnackbar({
        open: true,
        message: error.message || 'Failed to save dataset',
        severity: 'error'
      });
    } finally {
      setSaving(false);
    }
  };

  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      style={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100%',
      }}
    >
      {/* Page Header */}
      <Typography
        variant="h4"
        component="h1"
        gutterBottom
        className="cyber-header"
        sx={{ mb: 3 }}
      >
        Data Generation
      </Typography>

      {/* Main Content Area */}
      <Box sx={{
        display: 'flex',
        flexGrow: 1,
        gap: 3,
        width: '100%',
        minHeight: 'calc(100vh - 200px)',
        overflow: 'hidden',
      }}>
        {/* Main Card - Data Preview/Results Area */}
        <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <Card sx={{
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            border: '1px solid rgba(0, 230, 118, 0.2)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            overflow: 'hidden'
          }}>
            <CardContent sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              flexGrow: 1,
              overflow: 'auto'
            }}>
              <Typography variant="h6" sx={{ color: '#00E676', mb: 2, textAlign: 'center' }}>
                Data Preview & Generation Results
              </Typography>

              {/* Sample Dataset Selection - Show when no CSV is loaded */}
              {!hasCSV && !generatedData && !isGenerating && (
                <Box sx={{ mb: 3, maxWidth: '800px', mx: 'auto', width: '100%' }}>
                  <Alert severity="info" sx={{ mb: 3 }}>
                    <AlertTitle>No Dataset Loaded</AlertTitle>
                    Upload a CSV file in the Data Explorer, or select one of the sample datasets below to get started.
                  </Alert>

                  <Typography variant="h6" sx={{ color: '#00E676', mb: 2, textAlign: 'center' }}>
                    📊 Available Sample Datasets
                  </Typography>

                  {loadingSampleDatasets ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={20} sx={{ color: '#00E676' }} />
                      <Typography variant="body2">Loading datasets...</Typography>
                    </Box>
                  ) : (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {sampleDatasets.map((dataset) => (
                        <Card
                          key={dataset.filename}
                          sx={{
                            bgcolor: 'rgba(0, 230, 118, 0.05)',
                            border: '1px solid rgba(0, 230, 118, 0.2)',
                            borderRadius: 2,
                            transition: 'all 0.2s ease',
                            '&:hover': {
                              borderColor: '#00E676',
                              bgcolor: 'rgba(0, 230, 118, 0.1)',
                              transform: 'translateY(-2px)',
                              boxShadow: '0 4px 20px rgba(0, 230, 118, 0.2)',
                            }
                          }}
                        >
                          <CardContent sx={{ p: 2.5, '&:last-child': { pb: 2.5 } }}>
                            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, justifyContent: 'space-between', alignItems: { xs: 'stretch', sm: 'center' }, gap: 2 }}>
                              <Box sx={{ flex: 1, minWidth: 0 }}>
                                <Typography variant="h6" sx={{ color: '#00E676', display: 'flex', alignItems: 'center', gap: 1, fontSize: '1.1rem' }}>
                                  <DatasetIcon />
                                  {dataset.name}
                                </Typography>
                                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)', mt: 1, lineHeight: 1.5 }}>
                                  {dataset.description}
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1.5 }}>
                                  <Chip
                                    size="small"
                                    icon={<StorageIcon sx={{ fontSize: '16px !important' }} />}
                                    label={`${dataset.rows?.toLocaleString() || '?'} rows`}
                                    sx={{
                                      bgcolor: 'rgba(0, 230, 118, 0.15)',
                                      color: '#fff',
                                      fontWeight: 500,
                                      '& .MuiChip-icon': { color: '#00E676' }
                                    }}
                                  />
                                  {dataset.features && (
                                    <Chip
                                      size="small"
                                      label={`${dataset.features} features`}
                                      sx={{
                                        bgcolor: 'rgba(0, 230, 118, 0.15)',
                                        color: '#fff',
                                        fontWeight: 500
                                      }}
                                    />
                                  )}
                                </Box>
                              </Box>
                              <Button
                                variant="contained"
                                onClick={() => handleUseSampleDataset(dataset.filename)}
                                disabled={loadingDataset}
                                sx={{
                                  bgcolor: '#00E676',
                                  color: '#000',
                                  fontWeight: 600,
                                  minWidth: '120px',
                                  py: 1,
                                  '&:hover': {
                                    bgcolor: '#00C853',
                                  },
                                  '&.Mui-disabled': {
                                    bgcolor: 'rgba(0, 230, 118, 0.3)',
                                  }
                                }}
                              >
                                {loadingDataset && selectedDataset === dataset.filename ? (
                                  <CircularProgress size={20} sx={{ color: '#000' }} />
                                ) : (
                                  'Use Dataset'
                                )}
                              </Button>
                            </Box>
                          </CardContent>
                        </Card>
                      ))}
                    </Box>
                  )}
                </Box>
              )}

              {/* Current Dataset Info - Show when CSV is loaded */}
              {hasCSV && currentDatasetInfo && !isGenerating && (
                <Alert
                  severity="success"
                  sx={{ mb: 2 }}
                  icon={<DatasetIcon />}
                >
                  <AlertTitle>Dataset Loaded</AlertTitle>
                  <strong>{currentDatasetInfo.name || currentDatasetInfo.filename}</strong>
                  {currentDatasetInfo.rows && ` • ${currentDatasetInfo.rows.toLocaleString()} rows`}
                  {currentDatasetInfo.columns && ` • ${Array.isArray(currentDatasetInfo.columns) ? currentDatasetInfo.columns.length : currentDatasetInfo.columns} columns`}
                </Alert>
              )}

              {/* Persistent Progress Bar */}
              {isGenerating && (
                <Box sx={{ width: '100%', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                      Generating {numSamples} samples...
                    </Typography>
                    <Typography variant="subtitle2" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                      {Math.round(generationProgress)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={generationProgress}
                    sx={{
                      height: 8,
                      borderRadius: 2,
                      backgroundColor: 'rgba(0, 0, 0, 0.3)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: 'rgba(0, 230, 118, 0.8)',
                        borderRadius: 2,
                      }
                    }}
                  />
                </Box>
              )}

              {/* Generation Status and Error Messaging */}
              {isGenerating && (
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CircularProgress size={24} sx={{ color: '#00E676', mr: 2 }} />
                  <Typography>
                    Processing... this may take a few minutes for larger datasets
                  </Typography>
                </Box>
              )}

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  <AlertTitle>Error</AlertTitle>
                  {error}
                </Alert>
              )}

              {/* Data Table Display */}
              <Box
                sx={{
                  flexGrow: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: generatedData ? 'flex-start' : 'center',
                  alignItems: generatedData ? 'stretch' : 'center',
                  overflow: 'auto'
                }}
              >
                {!generatedData && !isGenerating ? (
                  <Typography variant="body1" color="text.secondary">
                    Configure parameters and click Generate Data to create synthetic data.
                  </Typography>
                ) : generatedData ? (
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Showing {generatedData.length} of {totalRows} rows
                      {isCombined && (
                        <span style={{ marginLeft: '8px', color: 'rgba(0, 230, 118, 0.8)' }}>
                          ({originalCount} original + {syntheticCount} synthetic)
                        </span>
                      )}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, justifyContent: 'space-between' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{
                          display: 'inline-block',
                          width: '12px',
                          height: '12px',
                          bgcolor: 'rgba(0, 0, 0, 0.3)',
                          mr: 0.5,
                          border: '1px solid rgba(255, 255, 255, 0.2)'
                        }} />
                        <Typography variant="caption" sx={{ mr: 2 }}>Original Data</Typography>

                        <Box sx={{
                          display: 'inline-block',
                          width: '12px',
                          height: '12px',
                          bgcolor: 'rgba(0, 230, 118, 0.15)',
                          mr: 0.5,
                          border: '1px solid rgba(0, 230, 118, 0.3)'
                        }} />
                        <Typography variant="caption">Synthetic Data</Typography>
                      </Box>

                      {isCombined && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => setShowCombined(!showCombined)}
                          sx={{
                            fontSize: '0.7rem',
                            py: 0.5,
                            color: 'rgba(0, 230, 118, 0.8)',
                            borderColor: 'rgba(0, 230, 118, 0.3)',
                            '&:hover': {
                              borderColor: 'rgba(0, 230, 118, 0.8)',
                              bgcolor: 'rgba(0, 230, 118, 0.05)'
                            }
                          }}
                        >
                          {showCombined ? 'Show Only Synthetic' : 'Show All Data'}
                        </Button>
                      )}
                    </Box>
                    <Paper
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden',
                        bgcolor: 'rgba(0, 0, 0, 0.2)',
                        border: '1px solid rgba(0, 230, 118, 0.1)',
                        minHeight: '300px',
                        maxHeight: '500px'
                      }}
                    >
                      <TableContainer sx={{
                        flexGrow: 1,
                        height: '100%',
                        overflow: 'auto'
                      }}>
                        <Table stickyHeader size="small">
                          <TableHead>
                            <TableRow>
                              {columns.map((column) => (
                                <TableCell
                                  key={column}
                                  sx={{
                                    bgcolor: 'rgba(0, 0, 0, 0.7)',
                                    color: '#00E676',
                                    fontWeight: 'bold'
                                  }}
                                >
                                  {column}
                                </TableCell>
                              ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {generatedData.map((row, rowIndex) => (
                              <TableRow
                                key={rowIndex}
                                sx={{
                                  '&:nth-of-type(odd)': {
                                    bgcolor: row.is_synthetic
                                      ? 'rgba(0, 230, 118, 0.15)'
                                      : 'rgba(0, 0, 0, 0.3)'
                                  },
                                  '&:nth-of-type(even)': {
                                    bgcolor: row.is_synthetic
                                      ? 'rgba(0, 230, 118, 0.1)'
                                      : 'rgba(0, 0, 0, 0.2)'
                                  },
                                  '&:hover': {
                                    bgcolor: row.is_synthetic
                                      ? 'rgba(0, 230, 118, 0.25)'
                                      : 'rgba(0, 230, 118, 0.1)'
                                  },
                                  borderLeft: row.is_synthetic
                                    ? '2px solid rgba(0, 230, 118, 0.5)'
                                    : 'none'
                                }}
                              >
                                {columns.map((column) => (
                                  <TableCell
                                    key={`${rowIndex}-${column}`}
                                    sx={{
                                      color: 'white',
                                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                                    }}
                                  >
                                    {row[column]}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Paper>

                    {/* Action Buttons - Download & Use for Analysis */}
                    <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 2, justifyContent: 'center' }}>
                      {/* Download Buttons */}
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          📥 Download Data
                        </Typography>
                        <ButtonGroup
                          variant="outlined"
                          size="small"
                          disabled={downloading}
                          sx={{
                            '& .MuiButton-root': {
                              color: '#00E676',
                              borderColor: 'rgba(0, 230, 118, 0.5)',
                              '&:hover': {
                                borderColor: '#00E676',
                                bgcolor: 'rgba(0, 230, 118, 0.1)'
                              }
                            }
                          }}
                        >
                          <Tooltip title="Download only the synthetic (generated) rows">
                            <Button
                              onClick={() => handleDownload('synthetic')}
                              startIcon={downloading ? <CircularProgress size={14} /> : <DownloadIcon />}
                            >
                              Synthetic Only
                            </Button>
                          </Tooltip>
                          {isCombined && (
                            <Tooltip title="Download original + synthetic data combined">
                              <Button
                                onClick={() => handleDownload('combined')}
                                startIcon={downloading ? <CircularProgress size={14} /> : <DownloadIcon />}
                              >
                                Combined Data
                              </Button>
                            </Tooltip>
                          )}
                        </ButtonGroup>
                      </Box>

                      <Divider orientation="vertical" flexItem sx={{ borderColor: 'rgba(0, 230, 118, 0.2)' }} />

                      {/* Use for Analysis Buttons */}
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          📊 Use for Analysis
                        </Typography>
                        <ButtonGroup
                          variant="contained"
                          size="small"
                          disabled={switchingData}
                          sx={{
                            '& .MuiButton-root': {
                              bgcolor: 'rgba(0, 230, 118, 0.2)',
                              color: '#00E676',
                              '&:hover': {
                                bgcolor: 'rgba(0, 230, 118, 0.3)'
                              }
                            }
                          }}
                        >
                          {isCombined && (
                            <Tooltip title="Switch to combined data for analysis & queries">
                              <Button
                                onClick={() => handleUseForAnalysis(true)}
                                startIcon={switchingData ? <CircularProgress size={14} /> : <AnalyticsIcon />}
                              >
                                Use Combined
                              </Button>
                            </Tooltip>
                          )}
                          <Tooltip title="Switch to synthetic data only for analysis & queries">
                            <Button
                              onClick={() => handleUseForAnalysis(false)}
                              startIcon={switchingData ? <CircularProgress size={14} /> : <AnalyticsIcon />}
                            >
                              Use Synthetic
                            </Button>
                          </Tooltip>
                        </ButtonGroup>
                      </Box>

                      <Divider orientation="vertical" flexItem sx={{ borderColor: 'rgba(0, 230, 118, 0.2)' }} />

                      {/* Save Dataset Button */}
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                          💾 Save Dataset
                        </Typography>
                        <Tooltip title="Save generated data for future use">
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => setSaveDialogOpen(true)}
                            startIcon={<SaveIcon />}
                            sx={{
                              color: '#FF9100',
                              borderColor: 'rgba(255, 145, 0, 0.5)',
                              '&:hover': {
                                borderColor: '#FF9100',
                                bgcolor: 'rgba(255, 145, 0, 0.1)'
                              }
                            }}
                          >
                            Save for Later
                          </Button>
                        </Tooltip>
                      </Box>
                    </Box>
                  </>
                ) : null}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Right Side Panel - Parameters */}
        <Box
          sx={{
            width: '260px',
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            borderRadius: '8px',
            p: 2,
            border: '1px solid rgba(0, 230, 118, 0.1)',
            alignSelf: 'flex-start',
            position: 'sticky',
            top: 16
          }}
        >
          <Typography variant="h6" sx={{ color: '#00E676', mb: 2, fontSize: '1rem' }}>
            Generation Parameters (Model: GPT-4o Mini)
          </Typography>

          {/* Generation Mode Toggle */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1, display: 'block' }}>
              Generation Mode
            </Typography>
            <ToggleButtonGroup
              value={generationMode}
              exclusive
              onChange={(e, newMode) => {
                if (newMode !== null) setGenerationMode(newMode);
              }}
              size="small"
              fullWidth
              disabled={isGenerating}
              sx={{
                '& .MuiToggleButton-root': {
                  color: 'rgba(255, 255, 255, 0.7)',
                  borderColor: 'rgba(0, 230, 118, 0.3)',
                  flex: 1,
                  py: 0.5,
                  '&.Mui-selected': {
                    color: '#00E676',
                    bgcolor: 'rgba(0, 230, 118, 0.15)',
                    borderColor: '#00E676',
                    '&:hover': {
                      bgcolor: 'rgba(0, 230, 118, 0.25)',
                    }
                  },
                  '&:hover': {
                    bgcolor: 'rgba(0, 230, 118, 0.1)',
                  }
                }
              }}
            >
              <Tooltip title="Fast: Single API call, great for most use cases (~5-10 seconds)">
                <ToggleButton value="fast">
                  <SpeedIcon sx={{ mr: 0.5, fontSize: '1rem' }} />
                  Fast
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Deep: Feature-by-feature generation with RAG context (slower but more precise)">
                <ToggleButton value="deep">
                  <PsychologyIcon sx={{ mr: 0.5, fontSize: '1rem' }} />
                  Deep
                </ToggleButton>
              </Tooltip>
            </ToggleButtonGroup>
            <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)', mt: 0.5, display: 'block', fontSize: '0.65rem' }}>
              {generationMode === 'fast' ? '⚡ Quick batch generation' : '🧠 Context-aware per-feature generation'}
            </Typography>
          </Box>

          <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)', mb: 1.5 }} />

          {/* Sidebar Progress Bar */}
          {isGenerating && (
            <Box sx={{ width: '100%', mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                  Progress
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(0, 230, 118, 0.9)' }}>
                  {Math.round(generationProgress)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={generationProgress}
                sx={{
                  height: 6,
                  borderRadius: 1,
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: 'rgba(0, 230, 118, 0.8)',
                    borderRadius: 1,
                  }
                }}
              />
              <Typography variant="caption" sx={{ display: 'block', mt: 1, textAlign: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
                Generating {numSamples} samples
              </Typography>
              <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)', my: 2 }} />
            </Box>
          )}

          <Stack spacing={1.5} sx={{ mb: 'auto' }}>
            {/* Number of Samples */}
            <TextField
              size="small"
              fullWidth
              label="Number of Samples"
              type="number"
              value={numSamples}
              onChange={(e) => {
                const value = Math.max(1, parseInt(e.target.value, 10) || 1);
                setNumSamples(value);
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(0, 230, 118, 0.3)' },
                  '&:hover fieldset': { borderColor: 'rgba(0, 230, 118, 0.5)' },
                  '&.Mui-focused fieldset': { borderColor: '#00E676' },
                },
                '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                '& .MuiInputBase-input': { color: 'white' },
              }}
              InputProps={{
                inputProps: { min: 1, max: 10000 }
              }}
              disabled={isGenerating}
            />

            <Divider sx={{ borderColor: 'rgba(0, 230, 118, 0.2)' }} />

            {/* Temperature */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Temperature</span>
                <span>{temperature}</span>
              </Typography>
              <Slider
                size="small"
                value={temperature}
                onChange={(e, newValue) => {
                  setTemperature(parseFloat(newValue.toFixed(1)));
                }}
                min={0.1}
                max={2.0}
                step={0.1}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>

            {/* Top-P */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Top-P</span>
                <span>{topP}</span>
              </Typography>
              <Slider
                size="small"
                value={topP}
                onChange={(e, newValue) => {
                  setTopP(parseFloat(newValue.toFixed(2)));
                }}
                min={0.1}
                max={1.0}
                step={0.05}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>

            {/* Frequency Penalty (renamed from Repetition Penalty) */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Frequency Penalty</span>
                <span>{repetitionPenalty}</span>
              </Typography>
              <Slider
                size="small"
                value={repetitionPenalty}
                onChange={(e, newValue) => {
                  setRepetitionPenalty(parseFloat(newValue.toFixed(1)));
                }}
                min={1.0}
                max={2.0}
                step={0.1}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>

            {/* Max Tokens */}
            <Box>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Max Tokens</span>
                <span>{maxTokens}</span>
              </Typography>
              <Slider
                size="small"
                value={maxTokens}
                onChange={(e, newValue) => {
                  setMaxTokens(parseInt(newValue, 10));
                }}
                min={256}
                max={4096}
                step={256}
                sx={{ color: '#00E676', py: 0, mt: 0.5 }}
                disabled={isGenerating}
              />
            </Box>
          </Stack>

          {/* Generate Button */}
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleGenerate}
            disabled={isGenerating}
            sx={{
              mt: 2,
              borderRadius: '4px',
              background: isGenerating ? 'rgba(0, 200, 83, 0.5)' : 'linear-gradient(90deg, #00C853, #00E676)',
              '&:hover': {
                background: isGenerating ? 'rgba(0, 200, 83, 0.5)' : 'linear-gradient(90deg, #00B34A, #00D26A)',
              },
              boxShadow: '0 2px 10px rgba(0, 230, 118, 0.3)'
            }}
            className="glow-effect"
            startIcon={isGenerating ? <CircularProgress size={16} sx={{ color: 'white' }} /> : null}
          >
            {isGenerating ? 'Generating...' : 'Generate Data'}
          </Button>
        </Box>
      </Box>

      {/* Save Dataset Dialog */}
      <Dialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
        PaperProps={{
          sx: {
            bgcolor: 'rgba(20, 20, 30, 0.95)',
            border: '1px solid rgba(0, 230, 118, 0.3)',
            borderRadius: 2,
            minWidth: 400
          }
        }}
      >
        <DialogTitle sx={{ color: '#00E676' }}>
          <SaveIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Save Generated Data
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: 'rgba(255,255,255,0.7)', mb: 2 }}>
            Save your generated data as a new dataset for future use. You can access it from the Datasets page.
          </DialogContentText>

          <TextField
            fullWidth
            label="Dataset Name"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="e.g., My Diabetes Study Data"
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(0, 230, 118, 0.3)' },
                '&:hover fieldset': { borderColor: 'rgba(0, 230, 118, 0.5)' },
                '&.Mui-focused fieldset': { borderColor: '#00E676' },
              },
              '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
              '& .MuiInputBase-input': { color: 'white' },
            }}
          />

          <TextField
            fullWidth
            label="Description (optional)"
            value={saveDescription}
            onChange={(e) => setSaveDescription(e.target.value)}
            placeholder="e.g., 100 synthetic rows generated from Pima dataset"
            multiline
            rows={2}
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(0, 230, 118, 0.3)' },
                '&:hover fieldset': { borderColor: 'rgba(0, 230, 118, 0.5)' },
                '&.Mui-focused fieldset': { borderColor: '#00E676' },
              },
              '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
              '& .MuiInputBase-input': { color: 'white' },
            }}
          />

          <Divider sx={{ my: 2, borderColor: 'rgba(0, 230, 118, 0.2)' }} />

          <Typography variant="subtitle2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 1 }}>
            What to save:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label="Synthetic Only"
              onClick={() => setSaveType('synthetic')}
              sx={{
                bgcolor: saveType === 'synthetic' ? 'rgba(0, 230, 118, 0.2)' : 'transparent',
                color: saveType === 'synthetic' ? '#00E676' : 'rgba(255,255,255,0.5)',
                border: `1px solid ${saveType === 'synthetic' ? '#00E676' : 'rgba(255,255,255,0.3)'}`,
                cursor: 'pointer'
              }}
            />
            {isCombined && (
              <Chip
                label="Combined (Original + Synthetic)"
                onClick={() => setSaveType('combined')}
                sx={{
                  bgcolor: saveType === 'combined' ? 'rgba(0, 230, 118, 0.2)' : 'transparent',
                  color: saveType === 'combined' ? '#00E676' : 'rgba(255,255,255,0.5)',
                  border: `1px solid ${saveType === 'combined' ? '#00E676' : 'rgba(255,255,255,0.3)'}`,
                  cursor: 'pointer'
                }}
              />
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)} sx={{ color: 'rgba(255,255,255,0.7)' }}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveDataset}
            disabled={saving || !saveName.trim()}
            startIcon={saving ? <CircularProgress size={16} /> : <SaveIcon />}
            sx={{ bgcolor: '#00E676', color: '#000' }}
          >
            {saving ? 'Saving...' : 'Save Dataset'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </motion.div>
  );
};

export default DataGeneration; 