import React, { useState } from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { Button, Card, TextInput, Paragraph } from 'react-native-paper';
import api, { setApiToken } from '../components/ApiClient';

// Helper to sanitize file names
const sanitizeFilename = (filename) => {
  filename = filename || 'synthetic_data.csv';
  // Extract filename and disallow path separators.
  const name = filename.replace(/.*[\\\/]/, '');
  // Basic validation: allowed characters and .csv only
  if (!/^[-_ a-zA-Z0-9]+\.csv$/.test(name)) {
    throw new Error('Invalid filename. Use alphanumerics, spaces, dashes and underscore and .csv extension');
  }
  return name;
}

export default function GenerateScreen() {
  const [nSamples, setNSamples] = useState('1000');
  const [outputFile, setOutputFile] = useState('synthetic_data.csv');
  const [running, setRunning] = useState(false);

  const handleGenerate = async () => {
    if (running) return;
    setRunning(true);
    try {
      // validation
      const n = Number(nSamples);
      if (!n || n < 10 || n > 20000) {
        Alert.alert('Invalid sample size', 'Must be between 10 and 20000');
        setRunning(false);
        return;
      }
      const safeName = sanitizeFilename(outputFile);

      const r = await api.post('/generate', { n_samples: n, output_file: safeName });
      if (r.status === 200) {
        Alert.alert('Success', `Generated ${n} samples to ${r.data.path || safeName}`);
      } else {
        Alert.alert('Failed', `Status ${r.status}`);
      }
    } catch (e) {
      Alert.alert('Error', e.message || 'Generation request failed');
    } finally {
      setRunning(false);
    }
  }

  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Card.Title title="Generate Synthetic Data" />
        <Card.Content>
          <Paragraph>Be mindful of sample size and output filename. API may limit generation for safety.</Paragraph>
          <TextInput label="n_samples" value={String(nSamples)} onChangeText={v => setNSamples(v)} keyboardType='numeric' />
          <TextInput label="output_file" value={outputFile} onChangeText={setOutputFile} />
          <Button mode="contained" style={{marginTop:10}} onPress={handleGenerate} loading={running} disabled={running}>
            Generate
          </Button>
        </Card.Content>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({ container: { padding: 12, flex: 1 }, card: { padding: 6 } });
