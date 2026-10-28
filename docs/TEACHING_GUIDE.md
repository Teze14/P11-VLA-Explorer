# A 15-minute VLA Explorer lesson

Audience: engineering students familiar with vectors and basic neural networks; researchers can use the low-level views to inspect the baseline's assumptions.

1. **Predict (2 min).** Keep the synthetic scene fixed. Ask: which region should score highest for a red cup? What changes if the instruction says blue cube?
2. **Observe (3 min).** Run each instruction and save snapshots. Open Similarity. Separate highest-scoring region from the weighted target: the latter may land in empty space.
3. **Explain (3 min).** Follow the five-stage trace. Students identify image/text embeddings, cosine scores, softmax weights, and the deterministic action. Ask which values are learned and which are computed by a chosen rule.
4. **Intervene (3 min).** Change opacity, temperature, and gain separately. Predict which values should stay unchanged before moving each slider.
5. **Challenge (2 min).** Try an absent object or occlude the strongest region. Ask why a softmax weight is not probability of correctness.
6. **Transfer (2 min).** Explain what a real robot-trained VLA must add: state, action semantics, training data, calibration/control, and evaluation.

## Answer guide

- Same image and grid: image embeddings can be cached. Different text requires its own text embedding.
- Opacity: no numerical model or action change. Temperature: changes weights and potentially target. Gain: changes delta, not target or embeddings.
- Weighted centers can lie between actual objects; heatmaps do not prove causality.
- The application runs a real pretrained vision-language model, but its action head is not a learned robotics policy.
- Camera snapshot is not continuous video inference. A 30 ms cached text update is not a 30 ms new-frame inference result.

## Para Jesus: explicacion breve en español

Piensa en cinco pasos: (1) dividir la imagen en regiones; (2) convertir cada region y el texto en vectores; (3) comparar los vectores; (4) convertir las comparaciones en pesos; (5) combinar los centros para proponer un punto. CLIP hace el trabajo aprendido de imagen/texto. Nosotros elegimos la regla que convierte los pesos en una flecha. El proyecto enseña esa diferencia y te deja experimentar con ella.

No necesitas defender que entrenaste un robot. Debes poder explicar qué calcula cada módulo, mostrar que el modelo sí responde a las entradas y describir con honestidad sus límites.
