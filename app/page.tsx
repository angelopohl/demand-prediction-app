"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Upload, Download, BarChart3, TrendingUp, FileText, AlertCircle } from "lucide-react"
import { ResultsDisplay } from "@/components/results-display"

export default function DemandPredictionApp() {
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile && selectedFile.type === "text/csv") {
      setFile(selectedFile)
      setError(null)
    } else {
      setError("Por favor selecciona un archivo CSV válido")
    }
  }

  const runAnalysis = async () => {
    if (!file) {
      setError("Por favor selecciona un archivo CSV")
      return
    }

    setIsAnalyzing(true)
    setProgress(0)
    setError(null)

    try {
      // Simular progreso del análisis
      const progressSteps = [
        { step: 20, message: "Cargando datos..." },
        { step: 40, message: "Preprocesando datos..." },
        { step: 60, message: "Entrenando modelos..." },
        { step: 80, message: "Generando predicciones..." },
        { step: 100, message: "Análisis completado" },
      ]

      for (const { step } of progressSteps) {
        await new Promise((resolve) => setTimeout(resolve, 1000))
        setProgress(step)
      }

      // Simular resultados del análisis
      const mockResults = {
        modelComparison: {
          LinearRegression: { MSE: 1234.56, R2: 0.7234 },
          DecisionTree: { MSE: 987.65, R2: 0.8123 },
          RandomForest: { MSE: 756.43, R2: 0.8567 },
        },
        bestModel: "RandomForest",
        featureImportance: [
          { feature: "Season_Code", importance: 0.35 },
          { feature: "Product_Category", importance: 0.28 },
          { feature: "Month", importance: 0.15 },
          { feature: "Warehouse_Name", importance: 0.12 },
          { feature: "Promo", importance: 0.1 },
        ],
        categoryAnalysis: [
          { category: "Electronics", avgDemand: 1250, error: 89 },
          { category: "Clothing", avgDemand: 980, error: 76 },
          { category: "Home & Garden", avgDemand: 750, error: 65 },
          { category: "Sports", avgDemand: 650, error: 58 },
        ],
      }

      setResults(mockResults)
    } catch (err) {
      setError("Error durante el análisis. Por favor intenta nuevamente.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const downloadResults = () => {
    // Simular descarga de archivos
    const csvContent =
      "data:text/csv;charset=utf-8,Category,Real,Prediction,Error\nElectronics,1200,1250,50\nClothing,950,980,30"
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "predicciones_por_categoria.csv")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center gap-2">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            Predicción de Demanda ML
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Analiza y predice la demanda de productos usando algoritmos de Machine Learning. Sube tu archivo CSV y obtén
            insights detallados sobre patrones de demanda.
          </p>
        </div>

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Cargar Datos
            </CardTitle>
            <CardDescription>
              Sube tu archivo CSV con datos de demanda. El archivo debe contener columnas como: Warehouse,
              Product_Category, Year, Month, Day, Weekday, Season, Promo, Order_Demand
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="csv-file">Archivo CSV</Label>
              <Input id="csv-file" type="file" accept=".csv" onChange={handleFileChange} disabled={isAnalyzing} />
            </div>

            {file && (
              <Alert>
                <FileText className="h-4 w-4" />
                <AlertDescription>
                  Archivo seleccionado: {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button onClick={runAnalysis} disabled={!file || isAnalyzing} className="w-full">
              {isAnalyzing ? "Analizando..." : "Ejecutar Análisis"}
            </Button>

            {isAnalyzing && (
              <div className="space-y-2">
                <Progress value={progress} className="w-full" />
                <p className="text-sm text-gray-600 text-center">Progreso: {progress}%</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {results && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <BarChart3 className="h-6 w-6" />
                Resultados del Análisis
              </h2>
              <Button onClick={downloadResults} variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Descargar Resultados
              </Button>
            </div>

            <ResultsDisplay results={results} />
          </div>
        )}
      </div>
    </div>
  )
}
