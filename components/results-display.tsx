"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart3, Target, TrendingUp, Award } from "lucide-react"

interface ResultsDisplayProps {
  results: {
    modelComparison: Record<string, { MSE: number; R2: number }>
    bestModel: string
    featureImportance: Array<{ feature: string; importance: number }>
    categoryAnalysis: Array<{ category: string; avgDemand: number; error: number }>
  }
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const { modelComparison, bestModel, featureImportance, categoryAnalysis } = results

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
              {Object.entries(modelComparison).map(([model, metrics]) => (
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
                        <span className="font-medium">{metrics.R2.toFixed(4)}</span>
                      </div>
                      <Progress value={metrics.R2 * 100} className="h-2" />
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-600">MSE: </span>
                      <span className="font-medium">{metrics.MSE.toFixed(2)}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
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
            <div className="space-y-4">
              {featureImportance.map((item, index) => (
                <div key={item.feature} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{item.feature}</span>
                    <span className="text-sm text-gray-600">{(item.importance * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={item.importance * 100} className="h-3" />
                </div>
              ))}
            </div>
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
            <div className="grid gap-4 md:grid-cols-2">
              {categoryAnalysis.map((category) => (
                <Card key={category.category}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">{category.category}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Demanda Promedio</span>
                      <span className="font-bold text-lg">{category.avgDemand.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Error Promedio</span>
                      <Badge variant="outline">{category.error}</Badge>
                    </div>
                    <Progress
                      value={(category.avgDemand / Math.max(...categoryAnalysis.map((c) => c.avgDemand))) * 100}
                      className="h-2"
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
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
              <p className="text-xs text-gray-600">R² = {modelComparison[bestModel].R2.toFixed(4)}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Característica Principal</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{featureImportance[0].feature}</div>
              <p className="text-xs text-gray-600">{(featureImportance[0].importance * 100).toFixed(1)}% importancia</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Categoría Top</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{categoryAnalysis[0].category}</div>
              <p className="text-xs text-gray-600">{categoryAnalysis[0].avgDemand.toLocaleString()} demanda promedio</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Precisión</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {(modelComparison[bestModel].R2 * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">R² Score</p>
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
                {modelComparison[bestModel].R2.toFixed(4)}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <p className="text-sm">
                <strong>Factor Clave:</strong> {featureImportance[0].feature} es la característica más importante para
                predecir la demanda
              </p>
            </div>
            <div className="p-3 bg-yellow-50 rounded-lg">
              <p className="text-sm">
                <strong>Oportunidad:</strong> {categoryAnalysis[0].category} tiene la mayor demanda promedio y podría
                beneficiarse de estrategias de inventario optimizadas
              </p>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}
