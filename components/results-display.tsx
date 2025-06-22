"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart3, Target, TrendingUp, Award } from "lucide-react"

interface DataInfo {
  totalSamples: number
  uniqueCategories: number
  uniqueWarehouses: number
  dateRange: string
  avgDemand: number
  maxDemand: number
  minDemand: number
}

interface ResultsDisplayProps {
  results: {
    modelComparison: Record<string, { MSE: number; R2: number }>
    bestModel: string
    featureImportance: Array<{ feature: string; importance: number }>
    categoryAnalysis: Array<{ category: string; avgDemand: number; error: number; count?: number }>
    dataInfo?: DataInfo
  }
}

// Función helper para formatear números de forma segura
const safeToFixed = (value: any, decimals = 2): string => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "N/A"
  }
  return Number(value).toFixed(decimals)
}

// Función helper para formatear números grandes
const safeToLocaleString = (value: any): string => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "N/A"
  }
  return Number(value).toLocaleString()
}

// Función helper para obtener valor seguro
const safeValue = (value: any, defaultValue: any = 0): any => {
  return value !== null && value !== undefined && !isNaN(Number(value)) ? value : defaultValue
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  // Validar que results existe y tiene la estructura esperada
  if (!results) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500">No hay resultados para mostrar</p>
        </CardContent>
      </Card>
    )
  }

  const { modelComparison = {}, bestModel = "N/A", featureImportance = [], categoryAnalysis = [], dataInfo } = results

  // Validar que tenemos datos mínimos
  if (Object.keys(modelComparison).length === 0) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500">Error: No se pudieron generar resultados del análisis</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Tabs defaultValue="models" className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="models">Modelos</TabsTrigger>
        <TabsTrigger value="features">Características</TabsTrigger>
        <TabsTrigger value="categories">Categorías</TabsTrigger>
        <TabsTrigger value="summary">Resumen</TabsTrigger>
      </TabsList>

      <TabsContent value="models" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Comparación de Modelos
            </CardTitle>
            <CardDescription>Rendimiento de diferentes algoritmos de Machine Learning</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(modelComparison).map(([model, metrics]) => {
                const r2Value = safeValue(metrics?.R2, 0)
                const mseValue = safeValue(metrics?.MSE, 0)

                return (
                  <Card key={model} className={model === bestModel ? "ring-2 ring-green-500" : ""}>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg flex items-center justify-between">
                        {model}
                        {model === bestModel && (
                          <Badge variant="default" className="bg-green-500">
                            <Award className="h-3 w-3 mr-1" />
                            Mejor
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div>
                        <div className="flex justify-between text-sm">
                          <span>R² Score</span>
                          <span className="font-medium">{safeToFixed(r2Value, 4)}</span>
                        </div>
                        <Progress value={Math.max(0, Math.min(100, r2Value * 100))} className="h-2" />
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-600">MSE: </span>
                        <span className="font-medium">{safeToFixed(mseValue, 2)}</span>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="features" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Importancia de Características
            </CardTitle>
            <CardDescription>Relevancia de cada variable en la predicción de demanda</CardDescription>
          </CardHeader>
          <CardContent>
            {featureImportance.length > 0 ? (
              <div className="space-y-4">
                {featureImportance.map((item, index) => {
                  const importance = safeValue(item?.importance, 0)
                  return (
                    <div key={`${item?.feature || "feature"}-${index}`} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{item?.feature || `Característica ${index + 1}`}</span>
                        <span className="text-sm text-gray-600">{safeToFixed(importance * 100, 1)}%</span>
                      </div>
                      <Progress value={Math.max(0, Math.min(100, importance * 100))} className="h-3" />
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-center text-gray-500">No hay datos de importancia de características disponibles</p>
            )}
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="categories" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Análisis por Categoría
            </CardTitle>
            <CardDescription>Demanda promedio predicha y error por categoría de producto</CardDescription>
          </CardHeader>
          <CardContent>
            {categoryAnalysis.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {categoryAnalysis.map((category, index) => {
                  const avgDemand = safeValue(category?.avgDemand, 0)
                  const error = safeValue(category?.error, 0)
                  const maxDemand = Math.max(...categoryAnalysis.map((c) => safeValue(c?.avgDemand, 0)))

                  return (
                    <Card key={`${category?.category || "category"}-${index}`}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">{category?.category || `Categoría ${index + 1}`}</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">Demanda Promedio</span>
                          <span className="font-bold text-lg">{safeToLocaleString(avgDemand)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">Error Promedio</span>
                          <Badge variant="outline">{safeToFixed(error, 1)}</Badge>
                        </div>
                        {category?.count && (
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Muestras</span>
                            <span className="text-sm font-medium">{category.count}</span>
                          </div>
                        )}
                        <Progress
                          value={maxDemand > 0 ? Math.max(0, Math.min(100, (avgDemand / maxDemand) * 100)) : 0}
                          className="h-2"
                        />
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            ) : (
              <p className="text-center text-gray-500">No hay datos de análisis por categoría disponibles</p>
            )}
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="summary" className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Mejor Modelo</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{bestModel}</div>
              <p className="text-xs text-gray-600">R² = {safeToFixed(modelComparison[bestModel]?.R2, 4)}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Total Muestras</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{safeToLocaleString(dataInfo?.totalSamples)}</div>
              <p className="text-xs text-gray-600">{dataInfo?.dateRange || "Rango de fechas"}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Categorías</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{safeValue(dataInfo?.uniqueCategories, categoryAnalysis.length)}</div>
              <p className="text-xs text-gray-600">Categorías únicas</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Demanda Promedio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {safeToLocaleString(Math.round(safeValue(dataInfo?.avgDemand, 0)))}
              </div>
              <p className="text-xs text-gray-600">Promedio general</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Recomendaciones</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="p-3 bg-blue-50 rounded-lg">
              <p className="text-sm">
                <strong>Modelo Recomendado:</strong> {bestModel} muestra el mejor rendimiento con un R² de{" "}
                {safeToFixed(modelComparison[bestModel]?.R2, 4)}
              </p>
            </div>
            {featureImportance.length > 0 && (
              <div className="p-3 bg-green-50 rounded-lg">
                <p className="text-sm">
                  <strong>Factor Clave:</strong> {featureImportance[0]?.feature || "N/A"} es la característica más
                  importante para predecir la demanda
                </p>
              </div>
            )}
            {categoryAnalysis.length > 0 && (
              <div className="p-3 bg-yellow-50 rounded-lg">
                <p className="text-sm">
                  <strong>Oportunidad:</strong> {categoryAnalysis[0]?.category || "N/A"} tiene la mayor demanda promedio
                  y podría beneficiarse de estrategias de inventario optimizadas
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}
