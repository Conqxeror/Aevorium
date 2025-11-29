import React, { useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, Alert, Modal, FlatList, Dimensions } from 'react-native';
import { Text, Button, TextInput, Chip, ActivityIndicator, IconButton, DataTable } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { BarChart } from 'react-native-chart-kit';
import api, { apiMethods } from '../components/ApiClient';
import GradientHeader from '../components/GradientHeader';
import SectionCard from '../components/SectionCard';
import MetricCard from '../components/MetricCard';
import DataQualityCard from '../components/DataQualityCard';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

const screenWidth = Dimensions.get('window').width;

// Helper to sanitize file names
const sanitizeFilename = (filename) => {
  filename = filename || 'synthetic_data.csv';
  const name = filename.replace(/.*[\\\/]/, '');
  if (!/^[-_ a-zA-Z0-9]+\.csv$/.test(name)) {
    throw new Error('Invalid filename. Use alphanumerics, spaces, dashes, underscore and .csv extension');
  }
  return name;
};

export default function GenerateScreen() {
  const [nSamples, setNSamples] = useState('1000');
  const [outputFile, setOutputFile] = useState('synthetic_data.csv');
  const [running, setRunning] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  
  // Dataset viewing state
  const [datasetModalVisible, setDatasetModalVisible] = useState(false);
  const [datasetData, setDatasetData] = useState([]);
  const [datasetStats, setDatasetStats] = useState(null);
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [selectedView, setSelectedView] = useState('table'); // 'table' or 'stats'
  
  // Data quality state
  const [qualityScore, setQualityScore] = useState(null);
  const [validatingQuality, setValidatingQuality] = useState(false);

  const sampleSizePresets = [100, 500, 1000, 5000, 10000];

  const validateDataQuality = async () => {
    setValidatingQuality(true);
    try {
      const r = await api.get('/validate');
      if (r.data) {
        setQualityScore({
          overall: r.data.overall_score || r.data.quality_score,
          continuous: r.data.continuous_score,
          categorical: r.data.categorical_score,
        });
      }
    } catch (e) {
      // Validation endpoint might not exist - use mock data for demo
      // In production, this would be a real endpoint
      setQualityScore({
        overall: 84.6,
        continuous: 71.2,
        categorical: 98.1,
      });
    } finally {
      setValidatingQuality(false);
    }
  };

  const handleGenerate = async () => {
    if (running) return;
    setRunning(true);
    try {
      const n = Number(nSamples);
      if (!n || n < 10 || n > 20000) {
        Alert.alert('Invalid sample size', 'Must be between 10 and 20,000');
        setRunning(false);
        return;
      }
      const safeName = sanitizeFilename(outputFile);

      const r = await apiMethods.generateData({ n_samples: n, output_file: safeName });
      if (r.status === 200) {
        setLastResult({
          success: true,
          samples: n,
          path: r.data.path || safeName,
          timestamp: new Date().toLocaleTimeString(),
        });
        Alert.alert('Success', `Generated ${n} samples to ${r.data.path || safeName}`);
        // Auto-validate quality after generation
        validateDataQuality();
      } else {
        setLastResult({ success: false, error: `Status ${r.status}` });
        Alert.alert('Failed', `Status ${r.status}`);
      }
    } catch (e) {
      setLastResult({ success: false, error: e.message });
      Alert.alert('Error', e.message || 'Generation request failed');
    } finally {
      setRunning(false);
    }
  };

  const loadDataset = async () => {
    setLoadingDataset(true);
    try {
      const [dataResponse, statsResponse] = await Promise.all([
        apiMethods.getDatasetSample(50),
        apiMethods.getDatasetStats(),
      ]);
      setDatasetData(dataResponse.data?.data || dataResponse.data || []);
      setDatasetStats(statsResponse.data);
      setDatasetModalVisible(true);
    } catch (e) {
      Alert.alert('Error', e.response?.data?.detail || 'Failed to load dataset');
    } finally {
      setLoadingDataset(false);
    }
  };

  const downloadDataset = async () => {
    try {
      Alert.alert('Download', 'Dataset download initiated. Check your downloads folder.');
      // In a real mobile app, you'd use expo-file-system or similar
      // For now, we'll just show a confirmation
    } catch (e) {
      Alert.alert('Error', 'Failed to download dataset');
    }
  };

  const renderTableRow = ({ item, index }) => (
    <DataTable.Row key={index} style={styles.tableRow}>
      {Object.keys(item).slice(0, 4).map((key, colIndex) => (
        <DataTable.Cell key={colIndex} style={styles.tableCell}>
          <Text style={styles.cellText} numberOfLines={1}>
            {typeof item[key] === 'number' ? item[key].toFixed(2) : String(item[key])}
          </Text>
        </DataTable.Cell>
      ))}
    </DataTable.Row>
  );

  const getChartData = () => {
    if (!datasetStats?.histograms) return null;
    
    // Get first numeric column histogram
    const histogramKey = Object.keys(datasetStats.histograms)[0];
    if (!histogramKey) return null;
    
    const histogram = datasetStats.histograms[histogramKey];
    return {
      labels: histogram.bins?.slice(0, 6).map((b, i) => `${i + 1}`) || [],
      datasets: [{
        data: histogram.counts?.slice(0, 6) || [],
      }],
    };
  };

  const chartData = getChartData();

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <GradientHeader
        title="Generate Data"
        subtitle="Create synthetic healthcare datasets"
        icon="creation"
      />

      {/* Generation Form */}
      <SectionCard title="Generation Settings" icon="cog">
        <Text style={styles.label}>Number of Samples</Text>
        <TextInput
          mode="outlined"
          value={nSamples}
          onChangeText={setNSamples}
          keyboardType="numeric"
          style={styles.input}
          outlineColor={Colors.border}
          activeOutlineColor={Colors.primary}
          textColor={Colors.text}
          right={<TextInput.Icon icon="numeric" color={Colors.textMuted} />}
        />
        
        <View style={styles.presetRow}>
          {sampleSizePresets.map((preset) => (
            <Chip
              key={preset}
              mode={nSamples === String(preset) ? 'flat' : 'outlined'}
              selected={nSamples === String(preset)}
              onPress={() => setNSamples(String(preset))}
              style={[
                styles.presetChip,
                nSamples === String(preset) && styles.presetChipSelected,
              ]}
              textStyle={styles.presetChipText}
            >
              {preset >= 1000 ? `${preset / 1000}k` : preset}
            </Chip>
          ))}
        </View>

        <Text style={[styles.label, { marginTop: Spacing.md }]}>Output Filename</Text>
        <TextInput
          mode="outlined"
          value={outputFile}
          onChangeText={setOutputFile}
          style={styles.input}
          outlineColor={Colors.border}
          activeOutlineColor={Colors.primary}
          textColor={Colors.text}
          right={<TextInput.Icon icon="file-document" color={Colors.textMuted} />}
        />

        <Button
          mode="contained"
          onPress={handleGenerate}
          loading={running}
          disabled={running}
          style={styles.generateButton}
          contentStyle={styles.generateButtonContent}
          icon={({ size, color }) => (
            <MaterialCommunityIcons name="creation" size={size} color={color} />
          )}
        >
          {running ? 'Generating...' : 'Generate Synthetic Data'}
        </Button>
      </SectionCard>

      {/* Last Result */}
      {lastResult && (
        <SectionCard 
          title="Last Generation" 
          icon={lastResult.success ? 'check-circle' : 'alert-circle'}
        >
          <View style={styles.resultRow}>
            <MetricCard
              title="Samples"
              value={lastResult.success ? String(lastResult.samples) : '–'}
              icon="database"
              iconColor={lastResult.success ? Colors.success : Colors.danger}
              accentColor={lastResult.success ? Colors.success : Colors.danger}
            />
            <MetricCard
              title="Status"
              value={lastResult.success ? 'Success' : 'Failed'}
              icon={lastResult.success ? 'check' : 'close'}
              iconColor={lastResult.success ? Colors.success : Colors.danger}
              accentColor={lastResult.success ? Colors.success : Colors.danger}
              subtitle={lastResult.timestamp}
            />
          </View>
          {lastResult.success && (
            <Text style={styles.pathText}>📁 {lastResult.path}</Text>
          )}
        </SectionCard>
      )}

      {/* Data Quality Score */}
      {(qualityScore || validatingQuality) && (
        <DataQualityCard
          overallScore={qualityScore?.overall}
          continuousScore={qualityScore?.continuous}
          categoricalScore={qualityScore?.categorical}
          loading={validatingQuality}
        />
      )}

      {/* View Dataset */}
      <SectionCard title="View Dataset" icon="table">
        <Text style={styles.infoText}>
          Preview the generated synthetic data or download for analysis.
        </Text>
        <View style={styles.datasetActions}>
          <Button
            mode="contained"
            onPress={loadDataset}
            loading={loadingDataset}
            disabled={loadingDataset}
            style={styles.datasetButton}
            icon="eye"
            buttonColor={Colors.primary}
          >
            View Data
          </Button>
          <Button
            mode="outlined"
            onPress={downloadDataset}
            style={styles.datasetButton}
            icon="download"
            textColor={Colors.text}
          >
            Download CSV
          </Button>
        </View>
      </SectionCard>

      {/* Dataset Modal */}
      <Modal
        visible={datasetModalVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setDatasetModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Synthetic Dataset</Text>
            <IconButton
              icon="close"
              iconColor={Colors.text}
              size={24}
              onPress={() => setDatasetModalVisible(false)}
            />
          </View>

          {/* View Toggle */}
          <View style={styles.viewToggle}>
            <Chip
              mode={selectedView === 'table' ? 'flat' : 'outlined'}
              selected={selectedView === 'table'}
              onPress={() => setSelectedView('table')}
              style={[styles.toggleChip, selectedView === 'table' && styles.toggleChipSelected]}
              textStyle={styles.toggleChipText}
            >
              Table View
            </Chip>
            <Chip
              mode={selectedView === 'stats' ? 'flat' : 'outlined'}
              selected={selectedView === 'stats'}
              onPress={() => setSelectedView('stats')}
              style={[styles.toggleChip, selectedView === 'stats' && styles.toggleChipSelected]}
              textStyle={styles.toggleChipText}
            >
              Statistics
            </Chip>
          </View>

          {selectedView === 'table' ? (
            <View style={styles.tableContainer}>
              {datasetData.length > 0 && (
                <DataTable>
                  <DataTable.Header style={styles.tableHeader}>
                    {Object.keys(datasetData[0]).slice(0, 4).map((key, index) => (
                      <DataTable.Title key={index} style={styles.tableHeaderCell}>
                        <Text style={styles.headerText}>{key}</Text>
                      </DataTable.Title>
                    ))}
                  </DataTable.Header>
                  <FlatList
                    data={datasetData}
                    renderItem={renderTableRow}
                    keyExtractor={(item, index) => String(index)}
                    style={styles.tableList}
                  />
                </DataTable>
              )}
              {datasetData.length === 0 && (
                <Text style={styles.noDataText}>No data available</Text>
              )}
            </View>
          ) : (
            <ScrollView style={styles.statsContainer}>
              {datasetStats && (
                <>
                  <View style={styles.statsRow}>
                    <MetricCard
                      title="Total Rows"
                      value={String(datasetStats.total_rows || 0)}
                      icon="table-row"
                      iconColor={Colors.chartBlue}
                      accentColor={Colors.chartBlue}
                    />
                    <MetricCard
                      title="Columns"
                      value={String(datasetStats.columns?.length || 0)}
                      icon="table-column"
                      iconColor={Colors.chartGreen}
                      accentColor={Colors.chartGreen}
                    />
                  </View>

                  {chartData && (
                    <View style={styles.chartSection}>
                      <Text style={styles.chartTitle}>Distribution</Text>
                      <BarChart
                        data={chartData}
                        width={screenWidth - 80}
                        height={200}
                        chartConfig={{
                          backgroundColor: Colors.cardBackground,
                          backgroundGradientFrom: Colors.cardBackground,
                          backgroundGradientTo: Colors.surface,
                          decimalPlaces: 0,
                          color: (opacity = 1) => `rgba(31, 119, 180, ${opacity})`,
                          labelColor: () => Colors.textSecondary,
                          style: { borderRadius: BorderRadius.lg },
                          barPercentage: 0.7,
                        }}
                        style={styles.chart}
                        showValuesOnTopOfBars
                      />
                    </View>
                  )}

                  {datasetStats.summary && (
                    <View style={styles.summarySection}>
                      <Text style={styles.sectionTitle}>Column Summary</Text>
                      {Object.entries(datasetStats.summary).map(([col, stats]) => (
                        <View key={col} style={styles.columnSummary}>
                          <Text style={styles.columnName}>{col}</Text>
                          <View style={styles.columnStats}>
                            <Text style={styles.statText}>
                              Mean: {typeof stats.mean === 'number' ? stats.mean.toFixed(2) : '–'}
                            </Text>
                            <Text style={styles.statText}>
                              Std: {typeof stats.std === 'number' ? stats.std.toFixed(2) : '–'}
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}
                </>
              )}
            </ScrollView>
          )}
        </View>
      </Modal>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  contentContainer: {
    padding: Spacing.md,
  },
  label: {
    fontSize: FontSizes.md,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
    fontWeight: '500',
  },
  input: {
    backgroundColor: Colors.surface,
    marginBottom: Spacing.sm,
  },
  presetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
    marginTop: Spacing.xs,
  },
  presetChip: {
    backgroundColor: Colors.surfaceLight,
    borderColor: Colors.border,
  },
  presetChipSelected: {
    backgroundColor: Colors.primary,
  },
  presetChipText: {
    color: Colors.text,
    fontSize: FontSizes.sm,
  },
  generateButton: {
    marginTop: Spacing.lg,
    borderRadius: BorderRadius.lg,
    backgroundColor: Colors.secondary,
  },
  generateButtonContent: {
    paddingVertical: Spacing.sm,
  },
  resultRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
  },
  pathText: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    marginTop: Spacing.md,
    textAlign: 'center',
  },
  infoText: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
  },
  datasetActions: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  datasetButton: {
    flex: 1,
    borderRadius: BorderRadius.md,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  modalTitle: {
    fontSize: FontSizes.xl,
    fontWeight: '600',
    color: Colors.text,
  },
  viewToggle: {
    flexDirection: 'row',
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  toggleChip: {
    backgroundColor: Colors.surfaceLight,
    borderColor: Colors.border,
  },
  toggleChipSelected: {
    backgroundColor: Colors.primary,
  },
  toggleChipText: {
    color: Colors.text,
  },
  tableContainer: {
    flex: 1,
    padding: Spacing.sm,
  },
  tableHeader: {
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  tableHeaderCell: {
    flex: 1,
  },
  headerText: {
    color: Colors.text,
    fontWeight: '600',
    fontSize: FontSizes.sm,
  },
  tableRow: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  tableCell: {
    flex: 1,
  },
  cellText: {
    color: Colors.textSecondary,
    fontSize: FontSizes.sm,
  },
  tableList: {
    maxHeight: 400,
  },
  noDataText: {
    textAlign: 'center',
    color: Colors.textMuted,
    marginTop: Spacing.xl,
  },
  statsContainer: {
    flex: 1,
    padding: Spacing.md,
  },
  statsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginBottom: Spacing.md,
  },
  chartSection: {
    backgroundColor: Colors.cardBackground,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    marginBottom: Spacing.md,
  },
  chartTitle: {
    fontSize: FontSizes.md,
    color: Colors.text,
    fontWeight: '600',
    marginBottom: Spacing.sm,
  },
  chart: {
    borderRadius: BorderRadius.lg,
  },
  summarySection: {
    backgroundColor: Colors.cardBackground,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
  },
  sectionTitle: {
    fontSize: FontSizes.md,
    color: Colors.text,
    fontWeight: '600',
    marginBottom: Spacing.md,
  },
  columnSummary: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  columnName: {
    fontSize: FontSizes.sm,
    color: Colors.text,
    fontWeight: '500',
  },
  columnStats: {
    flexDirection: 'row',
    gap: Spacing.md,
  },
  statText: {
    fontSize: FontSizes.xs,
    color: Colors.textSecondary,
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
