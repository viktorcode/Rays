//
//  GameViewController.swift
//  Rays macOS
//
//  Created by Viktor Chernikov on 16/04/2019.
//  Copyright © 2019 Viktor Chernikov. All rights reserved.
//

import Cocoa
import MetalKit

// Our macOS specific view controller
class GameViewController: NSViewController {

    var renderer: Renderer!
    var mtkView: MTKView!

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }

        guard let linearSRGB = CGColorSpace(name: CGColorSpace.linearSRGB) else {
            print("Linear SRGB colour space is not supported on this device")
            return
        }
        mtkView.colorspace = linearSRGB

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        mtkView.device = defaultDevice

        do {
            let newRenderer = try Renderer(withMetalKitView: mtkView)
            renderer = newRenderer
            renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
            mtkView.delegate = renderer
        } catch {
            print("Renderer cannot be initialized : \(error)")
        }
    }
}
