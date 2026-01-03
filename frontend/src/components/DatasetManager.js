import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import DatasetIcon from '@mui/icons-material/Dataset';
import DeleteIcon from '@mui/icons-material/Delete';
import DescriptionIcon from '@mui/icons-material/Description';
import RefreshIcon from '@mui/icons-material/Refresh';
import SaveIcon from '@mui/icons-material/Save';
import ScienceIcon from '@mui/icons-material/Science';
import StorageIcon from '@mui/icons-material/Storage';
import {
    Alert,
    Box,
    Button,
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
    IconButton,
    Paper,
    Snackbar,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tabs,
    TextField,
    Tooltip,
    Typography
} from '@mui/material';
import { motion } from 'framer-motion';
import { useContext, useEffect, useState } from 'react';
import { CSVContext, UploadResponseContext } from '../App';
import api from '../services/api';

const DatasetManager = () => {
    // State
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeDatasetId, setActiveDatasetId] = useState(null);
    const [activating, setActivating] = useState(null);
    const [error, setError] = useState(null);
    const [tabValue, setTabValue] = useState(0);
    const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

    // Preview state
    const [previewOpen, setPreviewOpen] = useState(false);
    const [previewData, setPreviewData] = useState(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewDataset, setPreviewDataset] = useState(null);

    // Save dialog state
    const [saveDialogOpen, setSaveDialogOpen] = useState(false);
    const [saveName, setSaveName] = useState('');
    const [saveDescription, setSaveDescription] = useState('');
    const [saveType, setSaveType] = useState('synthetic');
    const [saving, setSaving] = useState(false);

    // Delete dialog state
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [datasetToDelete, setDatasetToDelete] = useState(null);
    const [deleting, setDeleting] = useState(false);

    // Context
    const { setCSVUploaded } = useContext(CSVContext);
    const { setUploadResponse } = useContext(UploadResponseContext);

    // Load datasets on mount
    useEffect(() => {
        loadDatasets();
    }, []);

    const loadDatasets = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.getAllDatasets();
            if (response.success) {
                setDatasets(response.datasets || []);
                setActiveDatasetId(response.activeDatasetId);
            } else {
                throw new Error(response.error || 'Failed to load datasets');
            }
        } catch (err) {
            console.error('Error loading datasets:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleActivateDataset = async (dataset) => {
        setActivating(dataset.id);
        try {
            const response = await api.activateDataset(dataset.id);
            if (response.success) {
                setActiveDatasetId(dataset.id);
                setCSVUploaded(true);
                setUploadResponse(response);
                setSnackbar({
                    open: true,
                    message: `Activated: ${dataset.name}`,
                    severity: 'success'
                });
            } else {
                throw new Error(response.error);
            }
        } catch (err) {
            console.error('Error activating dataset:', err);
            setSnackbar({
                open: true,
                message: err.message || 'Failed to activate dataset',
                severity: 'error'
            });
        } finally {
            setActivating(null);
        }
    };

    const handlePreviewDataset = async (dataset) => {
        setPreviewDataset(dataset);
        setPreviewOpen(true);
        setPreviewLoading(true);
        try {
            const response = await api.previewDataset(dataset.id);
            if (response.success) {
                setPreviewData(response);
            } else {
                throw new Error(response.error);
            }
        } catch (err) {
            console.error('Error previewing dataset:', err);
            setSnackbar({
                open: true,
                message: err.message || 'Failed to preview dataset',
                severity: 'error'
            });
            setPreviewOpen(false);
        } finally {
            setPreviewLoading(false);
        }
    };

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
                    message: `Saved: ${saveName}`,
                    severity: 'success'
                });
                setSaveDialogOpen(false);
                setSaveName('');
                setSaveDescription('');
                loadDatasets(); // Refresh the list
            } else {
                throw new Error(response.error);
            }
        } catch (err) {
            console.error('Error saving dataset:', err);
            setSnackbar({
                open: true,
                message: err.message || 'Failed to save dataset',
                severity: 'error'
            });
        } finally {
            setSaving(false);
        }
    };

    const handleDeleteDataset = async () => {
        if (!datasetToDelete) return;

        setDeleting(true);
        try {
            const response = await api.deleteDataset(datasetToDelete.id);
            if (response.success) {
                setSnackbar({
                    open: true,
                    message: `Deleted: ${datasetToDelete.name}`,
                    severity: 'success'
                });
                setDeleteDialogOpen(false);
                setDatasetToDelete(null);
                loadDatasets(); // Refresh the list
            } else {
                throw new Error(response.error);
            }
        } catch (err) {
            console.error('Error deleting dataset:', err);
            setSnackbar({
                open: true,
                message: err.message || 'Failed to delete dataset',
                severity: 'error'
            });
        } finally {
            setDeleting(false);
        }
    };

    // Filter datasets by category
    const sampleDatasets = datasets.filter(d => d.category === 'sample');
    const savedDatasets = datasets.filter(d => d.category !== 'sample');

    const getCategoryColor = (category) => {
        switch (category) {
            case 'sample': return '#00E676';
            case 'generated_synthetic': return '#FF9100';
            case 'generated_combined': return '#00B0FF';
            default: return '#9E9E9E';
        }
    };

    const getCategoryLabel = (category) => {
        switch (category) {
            case 'sample': return 'Sample';
            case 'generated_synthetic': return 'Synthetic';
            case 'generated_combined': return 'Combined';
            case 'uploaded': return 'Uploaded';
            default: return 'Saved';
        }
    };

    const renderDatasetCard = (dataset) => (
        <Card
            key={dataset.id}
            sx={{
                bgcolor: activeDatasetId === dataset.id
                    ? 'rgba(0, 230, 118, 0.1)'
                    : 'rgba(0, 0, 0, 0.2)',
                border: activeDatasetId === dataset.id
                    ? '2px solid #00E676'
                    : '1px solid rgba(0, 230, 118, 0.2)',
                borderRadius: 2,
                transition: 'all 0.2s ease',
                '&:hover': {
                    borderColor: '#00E676',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 20px rgba(0, 230, 118, 0.2)',
                }
            }}
        >
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
                        {activeDatasetId === dataset.id && (
                            <CheckCircleIcon sx={{ color: '#00E676', fontSize: '1.2rem' }} />
                        )}
                        <Typography variant="h6" sx={{ color: '#00E676', fontSize: '1rem' }}>
                            {dataset.name}
                        </Typography>
                    </Box>
                    <Chip
                        size="small"
                        label={getCategoryLabel(dataset.category)}
                        sx={{
                            bgcolor: `${getCategoryColor(dataset.category)}22`,
                            color: getCategoryColor(dataset.category),
                            fontWeight: 600,
                            fontSize: '0.7rem'
                        }}
                    />
                </Box>

                {dataset.description && (
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 1.5, fontSize: '0.85rem' }}>
                        {dataset.description}
                    </Typography>
                )}

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1.5 }}>
                    <Chip
                        size="small"
                        icon={<StorageIcon sx={{ fontSize: '14px !important' }} />}
                        label={`${dataset.rows?.toLocaleString() || '?'} rows`}
                        sx={{
                            bgcolor: 'rgba(255, 255, 255, 0.1)',
                            color: '#fff',
                            fontSize: '0.75rem',
                            '& .MuiChip-icon': { color: 'rgba(255,255,255,0.7)' }
                        }}
                    />
                    {dataset.columns && (
                        <Chip
                            size="small"
                            label={`${Array.isArray(dataset.columns) ? dataset.columns.length : dataset.columns} columns`}
                            sx={{
                                bgcolor: 'rgba(255, 255, 255, 0.1)',
                                color: '#fff',
                                fontSize: '0.75rem'
                            }}
                        />
                    )}
                    {dataset.createdAt && (
                        <Chip
                            size="small"
                            label={dataset.createdAt}
                            sx={{
                                bgcolor: 'rgba(255, 255, 255, 0.05)',
                                color: 'rgba(255,255,255,0.6)',
                                fontSize: '0.7rem'
                            }}
                        />
                    )}
                </Box>

                <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                    <Tooltip title="Preview Data">
                        <IconButton
                            size="small"
                            onClick={() => handlePreviewDataset(dataset)}
                            sx={{ color: 'rgba(255,255,255,0.7)', '&:hover': { color: '#00E676' } }}
                        >
                            <DescriptionIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>

                    {dataset.canDelete && (
                        <Tooltip title="Delete Dataset">
                            <IconButton
                                size="small"
                                onClick={() => {
                                    setDatasetToDelete(dataset);
                                    setDeleteDialogOpen(true);
                                }}
                                sx={{ color: 'rgba(255,255,255,0.7)', '&:hover': { color: '#FF5252' } }}
                            >
                                <DeleteIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    )}

                    <Button
                        size="small"
                        variant={activeDatasetId === dataset.id ? "outlined" : "contained"}
                        onClick={() => handleActivateDataset(dataset)}
                        disabled={activating === dataset.id || activeDatasetId === dataset.id}
                        startIcon={activating === dataset.id ? <CircularProgress size={14} /> : null}
                        sx={{
                            bgcolor: activeDatasetId === dataset.id ? 'transparent' : '#00E676',
                            color: activeDatasetId === dataset.id ? '#00E676' : '#000',
                            borderColor: '#00E676',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                            '&:hover': {
                                bgcolor: activeDatasetId === dataset.id ? 'rgba(0,230,118,0.1)' : '#00C853',
                            },
                            '&.Mui-disabled': {
                                bgcolor: activeDatasetId === dataset.id ? 'transparent' : 'rgba(0, 230, 118, 0.3)',
                                color: activeDatasetId === dataset.id ? '#00E676' : 'rgba(0,0,0,0.5)',
                            }
                        }}
                    >
                        {activeDatasetId === dataset.id ? 'Active' : 'Activate'}
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            style={{ width: '100%' }}
        >
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" component="h1" className="cyber-header">
                    <DatasetIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Dataset Manager
                </Typography>

                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Save Generated Data">
                        <Button
                            variant="outlined"
                            startIcon={<SaveIcon />}
                            onClick={() => setSaveDialogOpen(true)}
                            sx={{
                                color: '#00E676',
                                borderColor: 'rgba(0, 230, 118, 0.5)',
                                '&:hover': { borderColor: '#00E676', bgcolor: 'rgba(0,230,118,0.1)' }
                            }}
                        >
                            Save Generated
                        </Button>
                    </Tooltip>

                    <Tooltip title="Refresh">
                        <IconButton
                            onClick={loadDatasets}
                            disabled={loading}
                            sx={{ color: '#00E676' }}
                        >
                            {loading ? <CircularProgress size={24} /> : <RefreshIcon />}
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            {/* Active Dataset Indicator */}
            {activeDatasetId && (
                <Paper sx={{
                    p: 2,
                    mb: 3,
                    bgcolor: 'rgba(0, 230, 118, 0.1)',
                    border: '1px solid rgba(0, 230, 118, 0.3)',
                    borderRadius: 2
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CheckCircleIcon sx={{ color: '#00E676' }} />
                        <Typography variant="subtitle1" sx={{ color: '#00E676', fontWeight: 600 }}>
                            Active Dataset: {datasets.find(d => d.id === activeDatasetId)?.name || 'Unknown'}
                        </Typography>
                    </Box>
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mt: 0.5 }}>
                        This dataset is ready for analysis and data generation.
                    </Typography>
                </Paper>
            )}

            {/* Tabs */}
            <Tabs
                value={tabValue}
                onChange={(e, newValue) => setTabValue(newValue)}
                sx={{
                    mb: 3,
                    '& .MuiTab-root': {
                        color: 'rgba(255,255,255,0.7)',
                        '&.Mui-selected': { color: '#00E676' }
                    },
                    '& .MuiTabs-indicator': { bgcolor: '#00E676' }
                }}
            >
                <Tab
                    icon={<ScienceIcon />}
                    iconPosition="start"
                    label={`Sample Datasets (${sampleDatasets.length})`}
                />
                <Tab
                    icon={<SaveIcon />}
                    iconPosition="start"
                    label={`Saved Datasets (${savedDatasets.length})`}
                />
            </Tabs>

            {/* Dataset Grid */}
            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress sx={{ color: '#00E676' }} />
                </Box>
            ) : (
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: 2 }}>
                    {tabValue === 0 && sampleDatasets.map(renderDatasetCard)}
                    {tabValue === 1 && (
                        savedDatasets.length > 0
                            ? savedDatasets.map(renderDatasetCard)
                            : (
                                <Paper sx={{
                                    p: 4,
                                    textAlign: 'center',
                                    bgcolor: 'rgba(0,0,0,0.2)',
                                    border: '1px dashed rgba(0, 230, 118, 0.3)',
                                    gridColumn: '1 / -1'
                                }}>
                                    <AddIcon sx={{ fontSize: 48, color: 'rgba(255,255,255,0.3)', mb: 2 }} />
                                    <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.5)', mb: 1 }}>
                                        No Saved Datasets Yet
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.4)', mb: 2 }}>
                                        Generate synthetic data and save it here for future use.
                                    </Typography>
                                    <Button
                                        variant="outlined"
                                        startIcon={<SaveIcon />}
                                        onClick={() => setSaveDialogOpen(true)}
                                        sx={{
                                            color: '#00E676',
                                            borderColor: 'rgba(0, 230, 118, 0.5)',
                                            '&:hover': { borderColor: '#00E676' }
                                        }}
                                    >
                                        Save Generated Data
                                    </Button>
                                </Paper>
                            )
                    )}
                </Box>
            )}

            {/* Preview Dialog */}
            <Dialog
                open={previewOpen}
                onClose={() => setPreviewOpen(false)}
                maxWidth="lg"
                fullWidth
                PaperProps={{
                    sx: {
                        bgcolor: 'rgba(20, 20, 30, 0.95)',
                        border: '1px solid rgba(0, 230, 118, 0.3)',
                        borderRadius: 2
                    }
                }}
            >
                <DialogTitle sx={{ color: '#00E676' }}>
                    Preview: {previewDataset?.name}
                </DialogTitle>
                <DialogContent>
                    {previewLoading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                            <CircularProgress sx={{ color: '#00E676' }} />
                        </Box>
                    ) : previewData ? (
                        <>
                            <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 2 }}>
                                Showing {previewData.previewRows} of {previewData.totalRows?.toLocaleString()} rows
                            </Typography>
                            <TableContainer sx={{ maxHeight: 400 }}>
                                <Table stickyHeader size="small">
                                    <TableHead>
                                        <TableRow>
                                            {previewData.columns?.map((col) => (
                                                <TableCell
                                                    key={col}
                                                    sx={{
                                                        bgcolor: 'rgba(0,0,0,0.8)',
                                                        color: '#00E676',
                                                        fontWeight: 'bold'
                                                    }}
                                                >
                                                    {col}
                                                </TableCell>
                                            ))}
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {previewData.data?.slice(0, 50).map((row, idx) => (
                                            <TableRow key={idx}>
                                                {previewData.columns?.map((col) => (
                                                    <TableCell
                                                        key={col}
                                                        sx={{
                                                            color: 'white',
                                                            borderBottom: '1px solid rgba(255,255,255,0.1)'
                                                        }}
                                                    >
                                                        {String(row[col])}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </>
                    ) : null}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setPreviewOpen(false)} sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        Close
                    </Button>
                    {previewDataset && activeDatasetId !== previewDataset.id && (
                        <Button
                            variant="contained"
                            onClick={() => {
                                handleActivateDataset(previewDataset);
                                setPreviewOpen(false);
                            }}
                            sx={{ bgcolor: '#00E676', color: '#000' }}
                        >
                            Activate This Dataset
                        </Button>
                    )}
                </DialogActions>
            </Dialog>

            {/* Save Dialog */}
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
                        Save your generated data as a new dataset for future use.
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

            {/* Delete Confirmation Dialog */}
            <Dialog
                open={deleteDialogOpen}
                onClose={() => setDeleteDialogOpen(false)}
                PaperProps={{
                    sx: {
                        bgcolor: 'rgba(20, 20, 30, 0.95)',
                        border: '1px solid rgba(255, 82, 82, 0.3)',
                        borderRadius: 2
                    }
                }}
            >
                <DialogTitle sx={{ color: '#FF5252' }}>
                    <DeleteIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Delete Dataset?
                </DialogTitle>
                <DialogContent>
                    <DialogContentText sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        Are you sure you want to delete <strong>{datasetToDelete?.name}</strong>?
                        This action cannot be undone.
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDeleteDialogOpen(false)} sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        onClick={handleDeleteDataset}
                        disabled={deleting}
                        startIcon={deleting ? <CircularProgress size={16} /> : <DeleteIcon />}
                        sx={{ bgcolor: '#FF5252', color: 'white' }}
                    >
                        {deleting ? 'Deleting...' : 'Delete'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Snackbar */}
            <Snackbar
                open={snackbar.open}
                autoHideDuration={4000}
                onClose={() => setSnackbar({ ...snackbar, open: false })}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert
                    onClose={() => setSnackbar({ ...snackbar, open: false })}
                    severity={snackbar.severity}
                    sx={{ width: '100%' }}
                >
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </motion.div>
    );
};

export default DatasetManager;
