//
//  GameViewController.swift
//  Rays macOS
//
//  Created by Viktor Chernikov on 16/04/2019.
//  Copyright © 2019 Viktor Chernikov. All rights reserved.
//

import Cocoa
import MetalKit

class NSLabel: NSTextField {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        self.isBezeled = false
        self.drawsBackground = false
        self.isEditable = false
        self.isSelectable = false
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

// Our macOS specific view controller
class GameViewController: NSViewController {

    var renderer: Renderer!
    var mtkView: MTKView!
    var counterView: NSLabel!

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

        counterView = NSLabel(frame: NSRect(x: 10, y: 10, width: 150, height: 50))
        counterView.stringValue = "----"
        counterView.textColor = .white
        mtkView.addSubview(counterView)

        do {
            let newRenderer = try Renderer(withMetalKitView: mtkView) { [unowned self] value in
                self.counterView.stringValue = String(format: "MRays/s: %.3f", value / 1_000_000)
            }
            renderer = newRenderer
            renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
            mtkView.delegate = renderer
        } catch {
            print("Renderer cannot be initialized : \(error)")
        }
    }
}
