@preconcurrency import CoreML
import Foundation

/// Inputs to the Unified encoder.
final class UnifiedEncoderFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["mel", "mel_length"]

    let mel: MLFeatureValue
    let melLength: MLFeatureValue

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "mel": return mel
        case "mel_length": return melLength
        default: return nil
        }
    }

    init(mel: MLMultiArray, melLength: MLMultiArray) {
        self.mel = MLFeatureValue(multiArray: mel)
        self.melLength = MLFeatureValue(multiArray: melLength)
    }
}

/// Inputs to the Unified RNNT decoder.
final class UnifiedDecoderFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["targets", "target_length", "h_in", "c_in"]

    let hIn: MLFeatureValue
    let cIn: MLFeatureValue
    let targets: MLFeatureValue
    let targetLength: MLFeatureValue

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "h_in": return hIn
        case "c_in": return cIn
        case "targets": return targets
        case "target_length": return targetLength
        default: return nil
        }
    }

    init(
        targets: MLMultiArray,
        targetLength: MLMultiArray,
        hIn: MLMultiArray,
        cIn: MLMultiArray
    ) {
        self.targets = MLFeatureValue(multiArray: targets)
        self.targetLength = MLFeatureValue(multiArray: targetLength)
        self.hIn = MLFeatureValue(multiArray: hIn)
        self.cIn = MLFeatureValue(multiArray: cIn)
    }
}

/// Inputs to the Unified joint decision model.
final class UnifiedJointDecisionFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["encoder_step", "decoder_step"]

    let encoderStep: MLFeatureValue
    let decoderStep: MLFeatureValue

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "encoder_step": return encoderStep
        case "decoder_step": return decoderStep
        default: return nil
        }
    }

    init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
        self.encoderStep = MLFeatureValue(multiArray: encoderStep)
        self.decoderStep = MLFeatureValue(multiArray: decoderStep)
    }
}
